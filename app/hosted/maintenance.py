"""Private backups, restore, and reconciliation for the hosted deployment.

Production snapshots use PostgreSQL's exported snapshot and SQLite's backup API.
A manifest is the commit marker: an incomplete backup never protects objects from
collection. Run only one maintenance service; the shared global database lock
also serializes manual maintenance with uploads, deletion, and other backups.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
from uuid import UUID, uuid4

from sqlalchemy import delete, inspect, select, text
from sqlalchemy.engine import make_url
from fastapi import HTTPException

from .config import Settings
from .uploads import upload_lock
from .database import Database
from .jobs import finalize_quota
from .models import Attempt, AuthFlow, AuthSession, ImageObject, Job, MetadataMutation, Project, UploadReservation, User, utcnow
from .quota import mutate_project, recover_project
from .storage import R2Store

LOG = logging.getLogger("annotation.maintenance")
RETENTION = timedelta(days=7)
ORPHAN_GRACE = timedelta(hours=24)
MANIFEST_VERSION = 1


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Object has no valid modification time")
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Invalid project or backup identity")
    return value


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _descriptor(key, data, **extra):
    return {"key": key, "sha256": _hash(data), "size_bytes": len(data), **extra}


def _thumbnail(key):
    return key.rsplit("/", 1)[0] + "/thumbnail.png"


def _verified(objects, descriptor):
    data = objects.get(descriptor["key"])
    if len(data) != descriptor["size_bytes"] or _hash(data) != descriptor["sha256"]:
        raise ValueError("Backup or image checksum mismatch")
    return data


def _validate_descriptor(item):
    if not isinstance(item, dict) or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
        raise ValueError("Invalid backup checksum")
    if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] < 0:
        raise ValueError("Invalid backup object size")


def load_manifest(settings, objects, key):
    """Validate all paths before permitting reads, restore, or retention changes."""
    parts = key.split("/")
    if len(parts) != 3 or parts[0] != "backups" or parts[2] != "manifest.json":
        raise ValueError("Expected a backup manifest key")
    backup_id = _uuid(parts[1])
    value = json.loads(objects.get(key))
    if value.get("version") != MANIFEST_VERSION or value.get("id") != backup_id:
        raise ValueError("Unsupported or inconsistent backup manifest")
    if value.get("environment") != settings.environment or value.get("bucket") != settings.r2_bucket:
        raise ValueError("Backup environment or bucket does not match configuration")
    _utc(value["created_at"])
    metadata = value["metadata"]
    _validate_descriptor(metadata)
    if metadata.get("format") not in {"postgresql-custom", "sqlite"}:
        raise ValueError("Unsupported metadata snapshot format")
    expected = "metadata.dump" if metadata["format"] == "postgresql-custom" else "metadata.sqlite3"
    if metadata["key"] != f"backups/{backup_id}/{expected}":
        raise ValueError("Invalid metadata snapshot key")
    project_ids = set()
    for project in value["projects"]:
        _validate_descriptor(project)
        project_id = _uuid(project["id"])
        if project_id in project_ids or project["key"] != f"backups/{backup_id}/projects/{project_id}.sqlite3":
            raise ValueError("Invalid project snapshot key")
        project_ids.add(project_id)
    object_keys = set()
    for item in value["objects"]:
        _validate_descriptor(item)
        components = item["key"].split("/")
        if len(components) != 5 or components[0] != "projects" or components[2] != "images":
            raise ValueError("Invalid original object key")
        _uuid(components[1]); _uuid(components[3])
        if components[1] not in project_ids or components[4] not in {"original", "thumbnail.png"}:
            raise ValueError("Invalid image object reference")
        if item["key"] in object_keys:
            raise ValueError("Duplicate image object reference")
        object_keys.add(item["key"])
    return value


def _pg_environment(database_url):
    """Keep passwords and connection URLs out of subprocess arguments and logs."""
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql" or not url.database:
        raise ValueError("A PostgreSQL target database is required")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
    for key, value in (("PGHOST", url.host), ("PGPORT", url.port), ("PGDATABASE", url.database),
                       ("PGUSER", url.username), ("PGPASSWORD", url.password)):
        if value is not None:
            env[key] = str(value)
    for key in ("sslmode", "sslrootcert", "sslcert", "sslkey", "connect_timeout"):
        if key in url.query:
            env["PG" + key.upper()] = str(url.query[key])
    env.setdefault("PGCONNECT_TIMEOUT", "10")
    return env


def _pg_tool(args, database_url):
    try:
        result = subprocess.run(args, env=_pg_environment(database_url), capture_output=True, check=False, timeout=600)
    except FileNotFoundError as error:
        raise RuntimeError("PostgreSQL backup utilities are not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("PostgreSQL maintenance command exceeded ten minutes") from error
    if result.returncode:
        # PostgreSQL stderr can contain connection details; do not log it.
        raise RuntimeError(f"PostgreSQL maintenance command failed (exit {result.returncode})")


def _snapshot_sqlite(source, destination):
    with closing(sqlite3.connect(str(destination))) as target:
        if isinstance(source, sqlite3.Connection):
            source.backup(target)
        else:
            with closing(sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
                connection.backup(target)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite backup integrity check failed")
    Path(destination).chmod(0o600)


@contextmanager
def _settled_projects(settings, db, objects):
    """Lock only committed, quota-accounted project generations for snapshots.

Recovery can replace SQLite and change metadata, so it must COMMIT before an
exported PostgreSQL snapshot. Recheck after reacquiring the global lock because
another request could have reserved a new generation in that interval.
    """
    while True:
        with db.session() as session:
            counter = db.global_lock(session)
            projects = list(session.scalars(select(Project).order_by(Project.id).with_for_update()))
            unsettled = [project.id for project in projects if project.pending_mutation_id or
                         (project.deleted_at is None and project.quota_version != 1)]
            if not unsettled:
                yield session, counter, projects
                return
        for project_id in unsettled:
            recover_project(settings, db, project_id, objects)


def _clean_pending_registrations(settings, db, objects, now):
    """Remove pre-journal crash residue through the same accounted mutation path."""
    with db.session() as session:
        candidates = list(session.execute(select(Project.id, Project.user_id).join(ImageObject).where(
            Project.deleted_at.is_(None), ImageObject.status == "pending",
            ImageObject.created_at < now - ORPHAN_GRACE).distinct()))
    for project_id, user_id in candidates:
        recover_project(settings, db, project_id, objects)
        def clean(store, session, project):
            image_ids = list(session.scalars(select(ImageObject.image_id).where(
                ImageObject.project_id == project.id, ImageObject.status == "pending",
                ImageObject.created_at < now - ORPHAN_GRACE)))
            with store._connect() as connection:
                for image_id in image_ids:
                    connection.execute("DELETE FROM annotation_sources WHERE internal_id IN (SELECT internal_id FROM coco_ids WHERE image_id=?)", (image_id,))
                    connection.execute("DELETE FROM coco_ids WHERE image_id=?", (image_id,))
                    connection.execute("DELETE FROM images WHERE id=?", (image_id,))
        try:
            mutate_project(settings, db, objects, project_id, user_id, clean, cleanup_only=True)
        except HTTPException as error:
            if error.status_code != 404:
                raise
            # Deletion won the race; the regular sweep collects that project.


def backup(settings, db, objects, now=None):
    """Create a consistent committed snapshot; originals remain immutable in R2."""
    now = _utc(now or utcnow())
    backup_id = str(uuid4())
    prefix = f"backups/{backup_id}"
    manifest = {"version": MANIFEST_VERSION, "id": backup_id,
                "created_at": now.isoformat() + "Z", "environment": settings.environment,
                "bucket": settings.r2_bucket, "projects": [], "objects": []}
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".backup-", dir=settings.data_dir) as temporary:
            directory = Path(temporary)
            with _settled_projects(settings, db, objects) as (session, _, projects):
                active = {project.id for project in projects if project.deleted_at is None}
                for project_id in sorted(active):
                    _uuid(project_id)
                    source = settings.data_dir / "projects" / project_id / "proyecto.sqlite3"
                    if not source.is_file():
                        raise FileNotFoundError("An active project's annotation database is missing")
                    destination = directory / f"{project_id}.sqlite3"
                    _snapshot_sqlite(source, destination)
                    data = destination.read_bytes()
                    key = f"{prefix}/projects/{project_id}.sqlite3"
                    objects.put(key, data)
                    manifest["projects"].append(_descriptor(key, data, id=project_id))
                images = list(session.scalars(select(ImageObject).where(
                    ImageObject.project_id.in_(active), ImageObject.status == "ready",
                    ImageObject.deleted_at.is_(None))))
                for image in images:
                    # UUID keys are immutable and the ready row is committed only
                    # after both writes succeed. No 50 GB daily object duplication.
                    manifest["objects"].append({"key": image.object_key, "sha256": image.sha256,
                                                "size_bytes": image.size_bytes})
                    thumbnail_key = _thumbnail(image.object_key)
                    thumb = objects.get(thumbnail_key)
                    manifest["objects"].append(_descriptor(thumbnail_key, thumb))
                if db.engine.dialect.name == "postgresql":
                    snapshot = session.scalar(text("SELECT pg_export_snapshot()"))
                    destination = directory / "metadata.dump"
                    _pg_tool(["pg_dump", "--format=custom", "--no-owner", "--no-acl",
                              f"--snapshot={snapshot}", f"--file={destination}"], settings.database_url)
                    metadata_format = "postgresql-custom"
                elif db.engine.dialect.name == "sqlite":
                    # This branch exists exclusively for injected offline tests.
                    destination = directory / "metadata.sqlite3"
                    _snapshot_sqlite(session.connection().connection.driver_connection, destination)
                    metadata_format = "sqlite"
                else:
                    raise ValueError("Unsupported metadata database")
                data = destination.read_bytes()
                metadata_key = f"{prefix}/{destination.name}"
                objects.put(metadata_key, data)
                manifest["metadata"] = _descriptor(metadata_key, data, format=metadata_format)
                manifest_key = f"{prefix}/manifest.json"
                objects.put(manifest_key, json.dumps(manifest, sort_keys=True).encode(), "application/json")
                # Keep global/project locks until the commit marker is durable:
                # reconciliation must not collect originals in this interval.
                return manifest_key
    except Exception:
        # Keep immutable artifacts after an ambiguous write outcome. If the
        # final manifest made it to R2 the snapshot is still usable; otherwise
        # reconciliation collects the partial prefix after 24 hours.
        raise


def _old(entry, cutoff):
    try:
        return _utc(entry["LastModified"]) < cutoff
    except (KeyError, ValueError, TypeError):
        return False  # Missing timestamps must never trigger deletion.


def _cleanup_local(settings, now, protected_mutations=(), protected_uploads=()):
    """Called under database locks; cache writes have their own shared lock."""
    cutoff = (now - ORPHAN_GRACE).replace(tzinfo=timezone.utc).timestamp()
    removed = 0
    if settings.data_dir.exists():
        for path in settings.data_dir.iterdir():
            if path.name.startswith((".backup-", "coco-")) and path.is_dir() and not path.is_symlink():
                latest = max((item.lstat().st_mtime for item in [path, *path.rglob("*")]), default=path.stat().st_mtime)
                if latest < cutoff:
                    shutil.rmtree(path)
                    removed += 1
    if settings.cache_dir.exists():
        with (settings.cache_dir / ".lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for path in settings.cache_dir.glob(".incoming-*"):
                if not path.is_symlink() and path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
    mutations = settings.data_dir / "mutations"
    if mutations.exists():
        for path in mutations.iterdir():
            if path.name in protected_mutations or path.is_symlink() or not path.is_dir():
                continue
            try:
                _uuid(path.name)
            except (ValueError, TypeError):
                continue
            latest = max(item.lstat().st_mtime for item in [path, *path.rglob("*")])
            if latest < cutoff:
                shutil.rmtree(path)
                removed += 1
    upload_dir = settings.data_dir / "uploads"
    if upload_dir.exists():
        for path in upload_dir.glob("*.lock"):
            if path.stem in protected_uploads or path.stat().st_mtime >= cutoff:
                continue
            try:
                with upload_lock(settings, path.stem, blocking=False) as data:
                    # Committed uploads can leave a part after a process crash.
                    data.unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
            except (BlockingIOError, ValueError):
                continue
    return removed


def reconcile(settings, db, objects, now=None):
    """Release abandoned uploads and collect only unreferenced, aged objects."""
    now = _utc(now or utcnow())
    result = {"expired_reservations": 0, "pending_images": 0, "deleted_objects": 0,
              "expired_backups": 0, "expired_auth": 0, "expired_jobs": 0,
              "expired_queued_jobs": 0, "local_artifacts": 0, "deleted_project_files": 0,
              "deleted_project_records": 0}
    _clean_pending_registrations(settings, db, objects, now)
    with _settled_projects(settings, db, objects) as (session, counter, projects):
        project_map = {project.id: project for project in projects}
        entries = list(objects.list())
        manifests = [(entry["Key"], load_manifest(settings, objects, entry["Key"]))
                     for entry in entries if entry["Key"].startswith("backups/") and entry["Key"].endswith("/manifest.json")]
        retained = [(key, value) for key, value in manifests if _utc(value["created_at"]) >= now - RETENTION]
        expired_prefixes = {key.rsplit("/", 1)[0] + "/" for key, value in manifests
                            if _utc(value["created_at"]) < now - RETENTION}
        retained_prefixes = {key.rsplit("/", 1)[0] + "/" for key, _ in retained}
        protected = {item["key"] for _, value in retained for item in value["objects"]}
        protected_projects = {item["id"] for _, value in retained for item in value["projects"]}
        for reservation in session.scalars(select(UploadReservation).where(UploadReservation.expires_at <= now).with_for_update()):
            if reservation.file_name:
                try:
                    with upload_lock(settings, reservation.id, blocking=False) as path:
                        path.unlink(missing_ok=True)
                except BlockingIOError:
                    continue
            user = session.get(User, reservation.user_id)
            if user:
                user.reserved_bytes = max(0, user.reserved_bytes - reservation.bytes)
            counter.reserved_bytes = max(0, counter.reserved_bytes - reservation.bytes)
            session.delete(reservation)
            result["expired_reservations"] += 1
        records = list(session.scalars(select(ImageObject)))
        referenced = set(protected)
        collectible = set()
        for image in records:
            project = project_map.get(image.project_id)
            old_pending = image.status == "pending" and image.created_at < now - ORPHAN_GRACE
            deleted = image.status == "deleted" or image.deleted_at is not None or project is None or project.deleted_at is not None
            keys = {image.object_key, _thumbnail(image.object_key)}
            if old_pending:
                session.delete(image)
                result["pending_images"] += 1
                collectible.update(keys - protected)
            elif deleted:
                collectible.update(keys - protected)
            else:
                referenced.update(keys)
        # Retained manifest references always win, including recently deleted
        # projects and originals needed to recover an older snapshot.
        for entry in entries:
            key = entry["Key"]
            delete_object = False
            if key in protected:
                continue
            if key in collectible:
                delete_object = True
            elif key.startswith(("exports/", "tmp/")):
                delete_object = _old(entry, now - ORPHAN_GRACE)
            elif key.startswith("projects/") and key not in referenced:
                delete_object = _old(entry, now - ORPHAN_GRACE)
            elif key.startswith("backups/"):
                prefix = "/".join(key.split("/")[:2]) + "/"
                delete_object = prefix in expired_prefixes or (prefix not in retained_prefixes and _old(entry, now - ORPHAN_GRACE))
            if delete_object:
                objects.delete(key)
                result["deleted_objects"] += 1
        result["expired_backups"] = len(expired_prefixes)
        for project in projects:
            if project.deleted_at is None or project.id in protected_projects:
                continue
            directory = settings.data_dir / "projects" / _uuid(project.id)
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)
                result["deleted_project_files"] += 1
        for job in session.scalars(select(Job).where(Job.status == "queued",
                                   Job.created_at <= now - ORPHAN_GRACE).with_for_update()):
            finalize_quota(session, job, charged=False)
            job.status, job.error, job.finished_at = "failed", "queue_expired", now
            job.payload, job.result = {}, None
            result["expired_queued_jobs"] += 1
        expired_jobs = select(Job.id).where(Job.status.in_(["succeeded", "failed", "cancelled"]),
                                            Job.finished_at <= now - ORPHAN_GRACE)
        # Finished payloads can contain reference images and dense masks. Quota
        # records are independent and deliberately survive job/result deletion.
        session.execute(delete(Attempt).where(Attempt.job_id.in_(expired_jobs)))
        deleted = session.execute(delete(Job).where(Job.id.in_(expired_jobs)))
        result["expired_jobs"] = deleted.rowcount or 0
        for model in (AuthFlow, AuthSession):
            deleted = session.execute(delete(model).where(model.expires_at <= now))
            result["expired_auth"] += deleted.rowcount or 0
        for project in projects:
            if project.deleted_at is None or project.id in protected_projects:
                continue
            if any(session.scalar(select(model.id).where(model.project_id == project.id).limit(1))
                   for model in (Job, UploadReservation, MetadataMutation)):
                continue
            # Usage was released when access was revoked. Remove filenames and
            # tombstones only after their backup and execution references end.
            session.execute(delete(ImageObject).where(ImageObject.project_id == project.id))
            session.delete(project)
            result["deleted_project_records"] += 1
        result["local_artifacts"] = _cleanup_local(settings, now, set(session.scalars(select(MetadataMutation.id))),
            set(session.scalars(select(UploadReservation.id))))
    return result


def restore(settings, objects, manifest_key, target_db, target_data_dir):
    """Restore into an empty, stopped deployment. All artifacts verify first."""
    target = Path(target_data_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError("Restore data directory must be empty")
    if inspect(target_db.engine).get_table_names():
        raise ValueError("Restore target database must be empty")
    manifest = load_manifest(settings, objects, manifest_key)
    metadata_format = manifest["metadata"]["format"]
    expected_format = "sqlite" if target_db.engine.dialect.name == "sqlite" else "postgresql-custom"
    if metadata_format != expected_format:
        raise ValueError("Snapshot format does not match target database")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=target.parent) as temporary:
        directory = Path(temporary)
        projects_path = directory / "projects"
        projects_path.mkdir(mode=0o700)
        metadata = directory / ("metadata.sqlite3" if metadata_format == "sqlite" else "metadata.dump")
        metadata.write_bytes(_verified(objects, manifest["metadata"]))
        metadata.chmod(0o600)
        for item in manifest["projects"]:
            destination = projects_path / item["id"] / "proyecto.sqlite3"
            destination.parent.mkdir(mode=0o700)
            destination.write_bytes(_verified(objects, item))
            destination.chmod(0o600)
            with closing(sqlite3.connect(destination)) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Project backup integrity check failed")
        for item in manifest["objects"]:
            _verified(objects, item)
        # Verify originals before changing either destination. R2 object keys
        # are shared with the source backup and cannot be restored from nothing.
        if metadata_format == "sqlite":
            with closing(sqlite3.connect(metadata)) as source, target_db.engine.connect() as destination:
                if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("Metadata backup integrity check failed")
                source.backup(destination.connection.driver_connection)
        else:
            url = target_db.engine.url.render_as_string(hide_password=False)
            _pg_tool(["pg_restore", "--no-owner", "--no-acl", "--exit-on-error", "--single-transaction",
                      f"--dbname={make_url(url).database}", str(metadata)], url)
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.replace(projects_path, target / "projects")
    with target_db.session() as session:
        target_db.global_lock(session)
        session.execute(delete(AuthSession))
        session.execute(delete(AuthFlow))
        for job in session.scalars(select(Job).where(Job.status.in_(["queued", "running"])).with_for_update()):
            job.status, job.error, job.finished_at = "failed", "Cancelled during backup restoration; submit a new request", utcnow()
            job.cancel_requested = True
            finalize_quota(session, job, charged=False)
    return {"projects": len(manifest["projects"]), "objects": len(manifest["objects"])}


def run(settings, db, objects, stop=None):
    stop = stop or threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    while not stop.is_set():
        try:
            reconcile(settings, db, objects)
            manifests = [load_manifest(settings, objects, item["Key"]) for item in objects.list("backups/")
                         if item["Key"].endswith("/manifest.json")]
            latest = max((_utc(item["created_at"]) for item in manifests), default=datetime.min)
            if utcnow() - latest >= timedelta(days=1):
                backup(settings, db, objects)
                LOG.info("Daily backup completed")
        except Exception as error:
            # Avoid logging exception text, which may contain SDK endpoints or
            # connection URLs. The service stays alive and retries in 5 minutes.
            LOG.error("Maintenance failed (%s); retry scheduled", type(error).__name__)
        stop.wait(300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("backup", "reconcile", "run"):
        commands.add_parser(command)
    recover = commands.add_parser("restore")
    recover.add_argument("--manifest", required=True, help="R2 backup manifest key")
    target_group = recover.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--database-url", help="Empty target PostgreSQL database URL; prefer --database-url-env to avoid shell history")
    recover.add_argument("--data-dir", required=True, type=Path, help="Empty target data directory")
    # An environment variable name permits keeping the URL out of process args.
    target_group.add_argument("--database-url-env", help="Read target database URL from this environment variable instead")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_env()
    settings.validate()
    objects = R2Store(settings)
    if args.command == "restore":
        from dataclasses import replace
        target_url = os.environ.get(args.database_url_env, "") if args.database_url_env else args.database_url
        if not target_url:
            parser.error("Restore target database URL is missing")
        target_settings = replace(settings, database_url=target_url, data_dir=args.data_dir)
        target_db = Database(target_settings)
        output = restore(settings, objects, args.manifest, target_db, args.data_dir)
    else:
        db = Database(settings)
        if args.command == "run":
            run(settings, db, objects)
            return
        output = backup(settings, db, objects) if args.command == "backup" else reconcile(settings, db, objects)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
