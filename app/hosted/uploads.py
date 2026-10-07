"""Quota-reserved, resumable image transfer in small authenticated requests."""
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import os
from pathlib import Path
import shutil
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select

from app.storage import _relative_file, IMAGE_EXTENSIONS
from .models import Project, UploadReservation, utcnow
from .storage import reserve_upload, release_reservation

CHUNK_BYTES = 8 * 1024 * 1024


def image_name(value):
    name = _relative_file(value)
    if len(name) > 1024 or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Use a supported image filename with a relative path")
    return name


def path_for(settings, upload_id):
    if str(UUID(upload_id)) != upload_id:
        raise HTTPException(404, "Upload not found")
    return settings.data_dir / "uploads" / f"{upload_id}.part"


@contextmanager
def upload_lock(settings, upload_id, *, blocking=True):
    path = path_for(settings, upload_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield path


def reservation_for(db, user_id, project_id, upload_id, *, refresh=False):
    with db.session() as session:
        db.global_lock(session)
        project = session.get(Project, project_id)
        row = session.get(UploadReservation, upload_id)
        if (not project or project.deleted_at or project.user_id != user_id or not row
                or row.user_id != user_id or row.project_id != project_id or not row.file_name
                or row.expires_at <= utcnow()):
            raise HTTPException(404, "Upload not found or expired")
        if refresh:
            row.expires_at = utcnow() + timedelta(hours=1)
        return row


def begin(settings, db, user_id, project_id, name, size):
    name = image_name(name)
    if shutil.disk_usage(settings.data_dir).free < settings.min_free_disk_bytes + size:
        raise HTTPException(503, "Server storage is temporarily unavailable")
    upload_id = reserve_upload(db, settings, user_id, project_id, size, file_name=name)
    return {"id": upload_id, "offset": 0, "chunk_bytes": CHUNK_BYTES}


def append(settings, db, user_id, project_id, upload_id, offset, data):
    reservation_for(db, user_id, project_id, upload_id)
    if not data or len(data) > CHUNK_BYTES:
        raise HTTPException(413, "Send the image in smaller chunks")
    with upload_lock(settings, upload_id) as path:
        row = reservation_for(db, user_id, project_id, upload_id, refresh=True)
        current = path.stat().st_size if path.exists() else 0
        if offset < 0 or offset > current or offset + len(data) > row.bytes:
            raise HTTPException(409, "Upload offset or reserved size does not match")
        if offset < current:
            # Lost response: replaying a durable chunk cannot duplicate bytes.
            with path.open("rb") as source:
                source.seek(offset)
                if source.read(len(data)) != data:
                    raise HTTPException(409, "Uploaded chunk differs; restart this upload")
            return {"offset": current}
        if shutil.disk_usage(settings.data_dir).free < settings.min_free_disk_bytes + len(data):
            raise HTTPException(503, "Server storage is temporarily unavailable")
        with path.open("ab") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {"offset": path.stat().st_size}


def abort(settings, db, user_id, project_id, upload_id):
    reservation_for(db, user_id, project_id, upload_id)
    with upload_lock(settings, upload_id) as path:
        reservation_for(db, user_id, project_id, upload_id)
        release_reservation(db, upload_id)
        path.unlink(missing_ok=True)
    return {"ok": True}
