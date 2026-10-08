"""Private object storage and a bounded, disposable, content-addressed cache."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import os
from pathlib import Path
import tempfile
import shutil

import boto3
from botocore.config import Config
from app.images import Image
from PIL import UnidentifiedImageError
from sqlalchemy import select

from app.storage import ProjectStore, _connection, _schema, _insert_image
from .models import ImageObject, Project, UploadReservation, User, utcnow, identity


class R2Store:
    def __init__(self, settings):
        self.bucket = settings.r2_bucket
        self.client = boto3.client("s3", endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id, aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto", config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=30,
                                             retries={"max_attempts": 3, "mode": "standard"}))

    def put(self, key, data, content_type="application/octet-stream"):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
                               CacheControl="private, no-store")

    def open_stream(self, key):
        from botocore.exceptions import ClientError
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                raise FileNotFoundError("Object not found") from exc
            raise

    def get(self, key):
        from botocore.exceptions import ClientError
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            with response["Body"] as stream:
                return stream.read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                raise FileNotFoundError("Object not found") from exc
            raise

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def list(self, prefix=""):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for entry in page.get("Contents", []):
                yield entry


class FilesystemStore:
    """Explicit injected adapter for offline tests; hosted deployment always uses R2."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Invalid object key")
        return path

    def put(self, key, data, content_type="application/octet-stream"):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(data, "read"):
            with path.open("wb") as output:
                shutil.copyfileobj(data, output)
        else:
            path.write_bytes(data)

    def open_stream(self, key):
        return self._path(key).open("rb")

    def get(self, key):
        return self._path(key).read_bytes()

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)

    def list(self, prefix=""):
        for path in self.root.rglob("*"):
            if path.is_file() and path.relative_to(self.root).as_posix().startswith(prefix):
                yield {"Key": path.relative_to(self.root).as_posix(), "Size": path.stat().st_size,
                       "LastModified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)}


class ObjectCache:
    def __init__(self, root, max_bytes):
        self.root, self.max_bytes = Path(root), max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, objects, key, sha256):
        # No user-supplied paths, and a shared filesystem lock protects eviction.
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("Invalid content hash")
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            path = self.root / sha256
            if path.exists():
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() == sha256:
                    path.touch()
                    return data
                path.unlink()
            data = objects.get(key)
            if hashlib.sha256(data).hexdigest() != sha256:
                raise ValueError("Stored image checksum mismatch")
            files = [p for p in self.root.iterdir() if p.is_file() and not p.name.startswith(".")]
            total = sum(p.stat().st_size for p in files)
            for old in sorted(files, key=lambda p: p.stat().st_mtime):
                if total + len(data) <= self.max_bytes:
                    break
                total -= old.stat().st_size
                old.unlink()
            if len(data) <= self.max_bytes:
                fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".incoming-")
                try:
                    with os.fdopen(fd, "wb") as output:
                        output.write(data)
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            return data


def project_store(settings, project_id):
    return ProjectStore(settings.data_dir / "projects" / project_id)


def create_project_store(settings, project_id, name):
    directory = settings.data_dir / "projects" / project_id
    directory.mkdir(parents=True, exist_ok=True)
    with _connection(directory / "proyecto.sqlite3") as connection:
        _schema(connection, directory, directory / "unavailable-originals")
        connection.execute("UPDATE meta SET value=? WHERE key='name'", (name,))
    return ProjectStore(directory)


def validate_image(source, settings=None):
    """Validate the complete original without imposing byte/pixel ceilings."""
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        stream.seek(0)
        with Image.open(stream) as image:
            if image.format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
                raise ValueError("Unsupported image format")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Only single still images are supported")
            result = (image.width, image.height, Image.MIME.get(image.format, "application/octet-stream"))
            image.verify()
        stream.seek(0)
        with Image.open(stream) as image:
            image.load()
        return result
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Invalid or damaged image") from exc
    finally:
        stream.seek(0)


def get_image_bytes(db, objects, project_id, image_id):
    with db.session() as session:
        image = session.execute(select(ImageObject).join(Project).where(
            ImageObject.project_id == project_id, ImageObject.image_id == image_id,
            ImageObject.status == "ready", Project.deleted_at.is_(None))).scalar_one_or_none()
        if image is None:
            raise FileNotFoundError("Image not found")
        data = objects.get(image.object_key)
        if hashlib.sha256(data).hexdigest() != image.sha256:
            raise ValueError("Stored image checksum mismatch")
        return data


def reserve_upload(db, settings, user_id, project_id, size, file_name=None):
    from fastapi import HTTPException
    with db.session() as session:
        counter = db.global_lock(session)
        user = session.get(User, user_id)
        project = session.get(Project, project_id)
        if project is None or project.user_id != user_id or project.deleted_at:
            raise HTTPException(404, "Project not found")
        if user.storage_bytes + user.reserved_bytes + size > settings.storage_limit_bytes:
            raise HTTPException(413, "Account storage quota exceeded")
        if counter.storage_bytes + counter.reserved_bytes + size > settings.global_storage_limit_bytes:
            raise HTTPException(503, "Storage capacity reached")
        user.reserved_bytes += size
        counter.reserved_bytes += size
        reservation = UploadReservation(id=identity(), user_id=user_id, project_id=project_id,
                                        bytes=size, file_name=file_name, expires_at=utcnow() + timedelta(hours=1))
        session.add(reservation)
        return reservation.id


def release_reservation(db, reservation_id):
    with db.session() as session:
        counter = db.global_lock(session)
        reservation = session.get(UploadReservation, reservation_id)
        if reservation:
            user = session.get(User, reservation.user_id)
            user.reserved_bytes -= reservation.bytes
            counter.reserved_bytes -= reservation.bytes
            session.delete(reservation)


def register_image(store, image):
    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        # Idempotence allows reconciliation after a crash between the two databases.
        row = connection.execute("SELECT sha256 FROM images WHERE id=?", (image.image_id,)).fetchone()
        if row:
            if row[0] != image.sha256:
                raise ValueError("Image identity conflict")
            return
        _insert_image(connection, image.image_id, image.file_name,
                      dict(width=image.width, height=image.height, sha256=image.sha256,
                           file_size=image.size_bytes, mtime_ns=0))
