import hashlib
import io
import json
import os
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image
import pytest
from sqlalchemy import create_engine, inspect, select

from app.hosted.config import Settings
from app.hosted.database import Database
from app.hosted.jobs import enqueue
from app.hosted.maintenance import backup, load_manifest, reconcile, restore, _pg_environment, _pg_tool
from app.hosted.models import AuthSession, ImageObject, InferenceUsage, Job, Project, ResourceCounter, UploadReservation, User, utcnow
from app.hosted.storage import FilesystemStore, create_project_store, project_store, register_image


@pytest.fixture
def setup(tmp_path):
    settings = Settings(database_url="unused", public_url="http://localhost:8765",
                        google_client_id="test", google_client_secret="test", r2_endpoint_url="unused",
                        r2_access_key_id="test", r2_secret_access_key="test", data_dir=tmp_path / "data",
                        cache_dir=tmp_path / "cache", min_free_disk_bytes=0)
    db = Database(settings, engine=create_engine(f"sqlite:///{tmp_path / 'source.sqlite3'}"))
    db.create_schema()
    objects = FilesystemStore(tmp_path / "objects")
    uid, pid, image_id = [str(uuid4()) for _ in range(3)]
    output = io.BytesIO()
    Image.new("RGB", (8, 6), "green").save(output, "PNG")
    data = output.getvalue()
    key = f"projects/{pid}/images/{image_id}/original"
    objects.put(key, data)
    objects.put(key.rsplit("/", 1)[0] + "/thumbnail.png", data)
    with db.session() as session:
        session.add(User(id=uid, google_sub="subject", email="owner@example.test", name="Owner", storage_bytes=len(data)))
        session.add(Project(id=pid, user_id=uid, name="Carrots"))
        record = ImageObject(id=image_id, project_id=pid, image_id=1, file_name="carrot.png", object_key=key,
                             sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), width=8, height=6,
                             content_type="image/png", status="ready")
        session.add(record)
        session.get(ResourceCounter, "global").storage_bytes = len(data)
        session.add(AuthSession(id="old-session", user_id=uid, csrf_token="old-csrf", expires_at=utcnow() + timedelta(days=2)))
    store = create_project_store(settings, pid, "Carrots")
    register_image(store, record)
    store.add_category("Carrot", "#3f513d")
    return SimpleNamespace(settings=settings, db=db, objects=objects, uid=uid, pid=pid, key=key, image_id=image_id,
                           store=store, data=data, root=tmp_path)


def target_for(value):
    target = Database(value.settings, engine=create_engine(f"sqlite:///{value.root / 'restored.sqlite3'}"))
    return target, value.root / "restored-data"


def seed_mixed_annotations(value):
    import numpy as np
    from app.geometry import mask_payload
    from app.hosted.quota import mutate_project
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 2:6] = True
    state = value.store.get_state(1)
    state.update(schema_version=2, annotations=[
        {"id": "manual-box", "kind": "bbox", "category_id": 1, "iscrowd": 0, "bbox": [1, 1, 5, 4]},
        {"id": "exact-mask", "category_id": 1, "iscrowd": 0, **mask_payload(mask)}],
        detection={"draft": {"id": "box-draft", "category_id": 1, "bbox": [0, 0, 2, 2], "points": []},
                   "proposals": [], "adjustment": None})
    return mutate_project(value.settings, value.db, value.objects, value.pid, value.uid,
                          lambda store, *_: store.save_state(1, state))


def age(objects, key, when):
    timestamp = when.replace(tzinfo=timezone.utc).timestamp()
    os.utime(objects._path(key), (timestamp, timestamp))


def test_backup_restore_consistent_projects_images_and_metadata(setup):
    value = setup
    mixed_state = seed_mixed_annotations(value)
    job = enqueue(value.db, value.settings, value.uid, value.pid, 1, "text", {"text": "carrot"})
    key = backup(value.settings, value.db, value.objects)
    manifest = load_manifest(value.settings, value.objects, key)
    assert len(manifest["projects"]) == 1
    assert {item["key"] for item in manifest["objects"]} == {value.key, value.key.rsplit("/", 1)[0] + "/thumbnail.png"}
    value.store.add_category("Later class", "#7856e8")
    with value.db.session() as session:
        session.get(Project, value.pid).deleted_at = utcnow()
    restored, target = target_for(value)
    assert restore(value.settings, value.objects, key, restored, target) == {"projects": 1, "objects": 2}
    with restored.session() as session:
        assert session.get(Project, value.pid).deleted_at is None
        assert session.get(User, value.uid).email == "owner@example.test"
        assert session.get(AuthSession, "old-session") is None
        assert session.get(Job, job["id"]).status == "failed"
        usage = session.scalar(select(InferenceUsage))
        assert usage.reserved == 0 and usage.used == 0
    restored_store = project_store(replace(value.settings, data_dir=target), value.pid)
    assert restored_store.get_state(1) == mixed_state
    project = restored_store.project()
    assert [item["name"] for item in project["categories"]] == ["Carrot"]
    assert project["images"][0]["file_name"] == "carrot.png"


def test_restore_refuses_nonempty_database_or_directory(setup):
    value = setup
    key = backup(value.settings, value.db, value.objects)
    empty_target = value.root / "empty-target"
    with pytest.raises(ValueError, match="database must be empty"):
        restore(value.settings, value.objects, key, value.db, empty_target)
    restored, target = target_for(value)
    target.mkdir(); (target / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="directory must be empty"):
        restore(value.settings, value.objects, key, restored, target)
    assert (target / "precious.txt").read_text() == "keep"


@pytest.mark.parametrize("corrupted", ["project", "original", "metadata"])
def test_restore_checks_every_artifact_before_mutation(setup, corrupted):
    value = setup
    key = backup(value.settings, value.db, value.objects)
    manifest = load_manifest(value.settings, value.objects, key)
    target_key = manifest["projects"][0]["key"] if corrupted == "project" else value.key if corrupted == "original" else manifest["metadata"]["key"]
    value.objects.put(target_key, b"corrupt")
    restored, target = target_for(value)
    with pytest.raises(ValueError, match="checksum"):
        restore(value.settings, value.objects, key, restored, target)
    assert not inspect(restored.engine).get_table_names()
    assert not target.exists()


def test_restore_rejects_traversal_and_different_environment(setup):
    value = setup
    key = backup(value.settings, value.db, value.objects)
    manifest = load_manifest(value.settings, value.objects, key)
    manifest["projects"][0]["id"] = "../../outside"
    value.objects.put(key, json.dumps(manifest).encode())
    with pytest.raises(ValueError):
        load_manifest(value.settings, value.objects, key)
    with pytest.raises(ValueError, match="environment"):
        load_manifest(replace(value.settings, environment="prod", r2_bucket="annotation-prod"), value.objects, key)


def test_deleted_originals_survive_until_last_referencing_backup_expires(setup):
    value = setup
    now = utcnow()
    key = backup(value.settings, value.db, value.objects, now=now)
    with value.db.session() as session:
        session.get(Project, value.pid).deleted_at = now
        record = session.get(ImageObject, value.image_id)
        record.status, record.deleted_at = "deleted", now
    reconcile(value.settings, value.db, value.objects, now=now + timedelta(days=6))
    assert value.objects.get(value.key) == value.data
    assert value.objects.get(key)
    local_project = value.settings.data_dir / "projects" / value.pid
    assert local_project.is_dir()
    result = reconcile(value.settings, value.db, value.objects, now=now + timedelta(days=8))
    assert result["expired_backups"] == 1
    assert result["deleted_project_files"] == 1
    assert result["deleted_project_records"] == 1
    assert not local_project.exists()
    with value.db.session() as session:
        assert session.get(Project, value.pid) is None
        assert session.get(ImageObject, value.image_id) is None
    with pytest.raises(FileNotFoundError):
        value.objects.get(value.key)
    assert not list(value.objects.list("backups/"))


def test_deleted_local_project_cleanup_does_not_follow_symlinks(setup):
    value = setup
    outside = value.root / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    deleted_id = str(uuid4())
    (value.settings.data_dir / "projects" / deleted_id).symlink_to(outside, target_is_directory=True)
    with value.db.session() as session:
        session.add(Project(id=deleted_id, user_id=value.uid, name="Deleted", deleted_at=utcnow()))
    result = reconcile(value.settings, value.db, value.objects)
    assert result["deleted_project_files"] == 0
    assert (outside / "keep.txt").read_text() == "keep"
    assert (value.settings.data_dir / "projects" / value.pid).is_dir()


def test_deleted_project_metadata_waits_for_active_jobs_before_purge(setup):
    from app.hosted.jobs import finalize_quota
    value = setup
    now = utcnow()
    job_id = enqueue(value.db, value.settings, value.uid, value.pid, 1, "text", {"text": "carrot"})["id"]
    with value.db.session() as session:
        session.get(Job, job_id).status = "running"
        session.get(Project, value.pid).deleted_at = now
        image = session.get(ImageObject, value.image_id)
        image.status, image.deleted_at = "deleted", now
    first = reconcile(value.settings, value.db, value.objects, now)
    assert first["deleted_project_records"] == 0
    with value.db.session() as session:
        job = session.get(Job, job_id)
        job.status, job.finished_at = "cancelled", now
        finalize_quota(session, job, charged=True)
    last = reconcile(value.settings, value.db, value.objects, now + timedelta(hours=25))
    assert last["expired_jobs"] == 1 and last["deleted_project_records"] == 1
    with value.db.session() as session:
        assert session.get(Project, value.pid) is None
        assert session.get(User, value.uid) is not None
        assert session.scalar(select(InferenceUsage)).used == 1


def test_reconciliation_releases_reservations_and_cleans_partial_uploads(setup):
    value = setup
    now = utcnow()
    pending_id = str(uuid4())
    pending_key = f"projects/{value.pid}/images/{pending_id}/original"
    value.objects.put(pending_key, value.data)
    with value.db.session() as session:
        user = session.get(User, value.uid)
        user.reserved_bytes = 100
        session.get(ResourceCounter, "global").reserved_bytes = 100
        session.add(UploadReservation(id=str(uuid4()), user_id=value.uid, project_id=value.pid, bytes=75, expires_at=now - timedelta(hours=1)))
        session.add(UploadReservation(id=str(uuid4()), user_id=value.uid, project_id=value.pid, bytes=25, expires_at=now + timedelta(hours=1)))
        pending = ImageObject(id=pending_id, project_id=value.pid, image_id=2, file_name="partial.png", object_key=pending_key,
                              sha256=hashlib.sha256(value.data).hexdigest(), size_bytes=len(value.data), width=8, height=6,
                              content_type="image/png", status="pending", created_at=now - timedelta(hours=25))
        session.add(pending)
    register_image(value.store, pending)
    result = reconcile(value.settings, value.db, value.objects, now)
    assert result["expired_reservations"] == 1 and result["pending_images"] == 1
    with value.db.session() as session:
        assert session.get(User, value.uid).reserved_bytes == 25
        assert session.get(ResourceCounter, "global").reserved_bytes == 25
        assert session.get(ImageObject, pending_id) is None
    assert len(value.store.project()["images"]) == 1
    with pytest.raises(FileNotFoundError):
        value.objects.get(pending_key)
    assert value.objects.get(value.key) == value.data


def test_orphan_sweep_respects_age_and_active_references(setup):
    value = setup
    now = utcnow()
    old = now - timedelta(hours=25)
    orphan = f"projects/{uuid4()}/images/{uuid4()}/original"
    fresh = f"projects/{uuid4()}/images/{uuid4()}/original"
    partial = f"backups/{uuid4()}/metadata.dump"
    for key in [orphan, fresh, partial, "exports/old.json", "tmp/old"]:
        value.objects.put(key, b"orphan")
        if key != fresh:
            age(value.objects, key, old)
    age(value.objects, value.key, old)
    result = reconcile(value.settings, value.db, value.objects, now)
    assert result["deleted_objects"] == 4
    assert value.objects.get(fresh) == b"orphan"
    assert value.objects.get(value.key) == value.data


def test_failed_manifest_write_never_creates_a_committed_snapshot(setup, monkeypatch):
    value = setup
    put = value.objects.put
    def fail_manifest(key, *args):
        if key.endswith("/manifest.json"):
            raise OSError("Storage unavailable")
        put(key, *args)
    monkeypatch.setattr(value.objects, "put", fail_manifest)
    with pytest.raises(OSError):
        backup(value.settings, value.db, value.objects)
    assert not any(entry["Key"].endswith("/manifest.json") for entry in value.objects.list())
    assert value.objects.get(value.key) == value.data


def test_corrupt_manifest_stops_collection_instead_of_losing_backup_references(setup):
    value = setup
    key = backup(value.settings, value.db, value.objects)
    with value.db.session() as session:
        record = session.get(ImageObject, value.image_id)
        record.status, record.deleted_at = "deleted", utcnow()
    value.objects.put(key, b"not-json")
    with pytest.raises(ValueError):
        reconcile(value.settings, value.db, value.objects)
    assert value.objects.get(value.key) == value.data


def test_pg_tools_keep_credentials_in_environment_and_redact_failures(monkeypatch):
    url = "postgresql+psycopg://backup:secret%40value@db.example.test:5433/annotation?sslmode=require&connect_timeout=4"
    env = _pg_environment(url)
    assert env["PGPASSWORD"] == "secret@value" and env["PGHOST"] == "db.example.test"
    assert env["PGSSLMODE"] == "require" and env["PGCONNECT_TIMEOUT"] == "4"
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stderr=b"contains secret@value")
    monkeypatch.setattr("app.hosted.maintenance.subprocess.run", run)
    with pytest.raises(RuntimeError, match="exit 1") as error:
        _pg_tool(["pg_dump", "--format=custom"], url)
    assert "secret" not in str(error.value)
    assert all("secret" not in argument for argument in calls[0][0])
    assert calls[0][1]["env"]["PGPASSWORD"] == "secret@value"


def test_local_crash_artifacts_expire_without_removing_active_files(setup):
    value = setup
    now = utcnow()
    timestamp = (now - timedelta(hours=25)).replace(tzinfo=timezone.utc).timestamp()
    for name in (".backup-crashed", "coco-crashed", ".backup-new"):
        directory = value.settings.data_dir / name
        directory.mkdir()
        file = directory / "partial"
        file.write_bytes(b"temporary")
        if name != ".backup-new":
            os.utime(file, (timestamp, timestamp))
            os.utime(directory, (timestamp, timestamp))
    value.settings.cache_dir.mkdir()
    cache = value.settings.cache_dir / ".incoming-crashed"
    cache.write_bytes(b"partial")
    os.utime(cache, (timestamp, timestamp))
    healthy = value.settings.cache_dir / "healthy-cache"
    healthy.write_bytes(b"keep")
    os.utime(healthy, (timestamp, timestamp))
    result = reconcile(value.settings, value.db, value.objects, now)
    assert result["local_artifacts"] == 3
    assert (value.settings.data_dir / ".backup-new" / "partial").exists()
    assert healthy.exists()
    assert (value.settings.data_dir / "projects" / value.pid / "proyecto.sqlite3").exists()


def test_backup_recovers_reserved_generation_before_taking_metadata_snapshot(setup, monkeypatch):
    from app.hosted import quota
    from app.hosted.models import MetadataMutation
    value = setup
    quota.recover_project(value.settings, value.db, value.pid, value.objects)
    with monkeypatch.context() as fault:
        def interrupted(*args, **kwargs):
            raise OSError("Process interrupted after journal commit")
        fault.setattr(quota, "recover_project", interrupted)
        with pytest.raises(OSError):
            quota.mutate_project(value.settings, value.db, value.objects, value.pid, value.uid,
                                 lambda store, session, project: store.add_category("Accepted before crash", "#7856e8"))
    with value.db.session() as session:
        assert session.get(Project, value.pid).pending_mutation_id
        assert session.get(User, value.uid).reserved_bytes > 0
    key = backup(value.settings, value.db, value.objects)
    restored, target = target_for(value)
    restore(value.settings, value.objects, key, restored, target)
    saved = project_store(replace(value.settings, data_dir=target), value.pid)
    assert [item["name"] for item in saved.project()["categories"]] == ["Carrot", "Accepted before crash"]
    with restored.session() as session:
        project = session.get(Project, value.pid)
        user = session.get(User, value.uid)
        assert project.pending_mutation_id is None and not list(session.scalars(select(MetadataMutation)))
        assert project.metadata_bytes == quota.logical_metadata_bytes(target / "projects" / value.pid / "proyecto.sqlite3")
        assert user.storage_bytes == len(value.data) * 2 + project.metadata_bytes
        assert user.reserved_bytes == 0


def test_unrecoverable_reserved_generation_stops_cleanup_and_preserves_artifacts(setup, monkeypatch):
    from app.hosted import quota
    value = setup
    now = utcnow()
    quota.recover_project(value.settings, value.db, value.pid, value.objects)
    with monkeypatch.context() as fault:
        def interrupted(*args, **kwargs):
            raise OSError("Interrupted")
        fault.setattr(quota, "recover_project", interrupted)
        with pytest.raises(OSError):
            quota.mutate_project(value.settings, value.db, value.objects, value.pid, value.uid,
                                 lambda store, session, project: store.add_category("Reserved", "#7856e8"))
    with value.db.session() as session:
        mutation_id = session.get(Project, value.pid).pending_mutation_id
        reserved = session.get(User, value.uid).reserved_bytes
    staged = quota.mutation_directory(value.settings, mutation_id) / "proyecto.sqlite3"
    staged.write_bytes(b"corrupt stage")
    old = (now - timedelta(hours=25)).replace(tzinfo=timezone.utc).timestamp()
    os.utime(staged, (old, old)); os.utime(staged.parent, (old, old))
    orphan = "tmp/old"
    value.objects.put(orphan, b"keep until recovery")
    age(value.objects, orphan, now - timedelta(hours=25))
    with pytest.raises(RuntimeError, match="generation is unavailable"):
        reconcile(value.settings, value.db, value.objects, now)
    assert staged.exists() and value.objects.get(orphan)
    with value.db.session() as session:
        assert session.get(User, value.uid).reserved_bytes == reserved


def test_orphaned_staging_generations_expire_after_one_day(setup):
    from app.hosted.quota import mutation_directory
    value = setup
    now = utcnow()
    old = (now - timedelta(hours=25)).replace(tzinfo=timezone.utc).timestamp()
    stale = mutation_directory(value.settings, str(uuid4()))
    fresh = mutation_directory(value.settings, str(uuid4()))
    for directory in (stale, fresh):
        directory.mkdir(parents=True)
        (directory / "proyecto.sqlite3").write_bytes(b"uncommitted stage")
    os.utime(stale / "proyecto.sqlite3", (old, old)); os.utime(stale, (old, old))
    result = reconcile(value.settings, value.db, value.objects, now)
    assert result["local_artifacts"] == 1 and not stale.exists() and fresh.exists()


def test_postgres_exported_snapshot_roundtrip(setup, monkeypatch):
    """Opt-in integration against an explicitly supplied disposable PG instance.

    ANNOTATION_TEST_POSTGRES_URL_FILE holds the admin DSN outside this repo.
    Matching-version pg_dump and pg_restore must be available on PATH. Only
    randomly named databases created by this test are dropped afterwards.
    """
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from app.hosted.models import Base
    from concurrent.futures import ThreadPoolExecutor
    import threading

    dsn_file = os.environ.get("ANNOTATION_TEST_POSTGRES_URL_FILE")
    if not dsn_file:
        pytest.skip("No isolated PostgreSQL test connection configured")
    mixed_state = seed_mixed_annotations(setup)
    url = make_url(Path(dsn_file).read_text().strip())
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    names = ["annotation_maintenance_" + uuid4().hex for _ in range(2)]
    engines = []
    created = []
    try:
        with admin.connect() as connection:
            for name in names:
                connection.execute(text(f'CREATE DATABASE "{name}"'))
                created.append(name)
        engines = [create_engine(url.set(database=name)) for name in names]
        settings = replace(setup.settings, database_url=url.set(database=names[0]).render_as_string(hide_password=False))
        source = Database(settings, engine=engines[0])
        Base.metadata.create_all(source.engine)
        with setup.db.engine.connect() as original, source.engine.begin() as destination:
            for table in Base.metadata.sorted_tables:
                rows = [dict(row) for row in original.execute(select(table)).mappings()]
                if rows:
                    destination.execute(table.insert(), rows)
        waiting = threading.Event()
        original_lock = source.global_lock
        def observed_lock(session):
            waiting.set()
            return original_lock(session)
        # Force the snapshot to wait for an existing writer. Both databases
        # must reflect the writer's committed version when that lock is free.
        with ThreadPoolExecutor(max_workers=1) as pool:
            with source.session() as writer:
                original_lock(writer)
                monkeypatch.setattr(source, "global_lock", observed_lock)
                future = pool.submit(backup, settings, source, setup.objects)
                assert waiting.wait(5)
                assert not future.done()
                writer.get(Project, setup.pid).name = "Carrots updated"
                with setup.store._connect() as annotations:
                    annotations.execute("UPDATE meta SET value=? WHERE key='name'", ("Carrots updated",))
                setup.store.add_category("Committed before snapshot", "#7856e8")
            key = future.result(timeout=60)
        manifest = load_manifest(settings, setup.objects, key)
        assert manifest["metadata"]["format"] == "postgresql-custom"
        target_db = Database(settings, engine=engines[1])
        destination_dir = setup.root / "pg-restored-data"
        assert restore(settings, setup.objects, key, target_db, destination_dir) == {"projects": 1, "objects": 2}
        with target_db.session() as session:
            assert session.get(Project, setup.pid).name == "Carrots updated"
            assert session.get(User, setup.uid).email == "owner@example.test"
            assert session.get(AuthSession, "old-session") is None
        restored_store = project_store(replace(settings, data_dir=destination_dir), setup.pid)
        assert restored_store.get_state(1) == mixed_state
        project = restored_store.project()
        assert project["name"] == "Carrots updated"
        assert [item["name"] for item in project["categories"]] == ["Carrot", "Committed before snapshot"]
        assert project["images"][0]["file_name"] == "carrot.png"
    finally:
        for engine in engines:
            engine.dispose()
        with admin.connect() as connection:
            for name in created:
                connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_terminal_job_retention_keeps_usage_and_live_jobs(setup):
    from app.hosted.jobs import finalize_quota
    from app.hosted.models import Attempt
    value = setup
    now = utcnow()
    terminal_ids = []
    for status, age_hours in (("succeeded", 25), ("failed", 25), ("cancelled", 25), ("succeeded", 1)):
        job_id = enqueue(value.db, value.settings, value.uid, value.pid, 1, "text", {"text": "reference"})["id"]
        terminal_ids.append(job_id)
        with value.db.session() as session:
            job = session.get(Job, job_id)
            job.status, job.finished_at = status, now - timedelta(hours=age_hours)
            job.result = {"large": "mask-result"}
            finalize_quota(session, job, charged=status == "succeeded")
            session.add(Attempt(id=str(uuid4()), job_id=job_id, worker_name="primary", status=status,
                                started_at=job.finished_at - timedelta(seconds=1), deadline_at=job.finished_at,
                                finished_at=job.finished_at))
    running = enqueue(value.db, value.settings, value.uid, value.pid, 1, "text", {"text": "running"})["id"]
    with value.db.session() as session:
        session.get(Job, running).status = "running"
    queued = enqueue(value.db, value.settings, value.uid, value.pid, 1, "text", {"text": "queued"})["id"]
    result = reconcile(value.settings, value.db, value.objects, now)
    assert result["expired_jobs"] == 3
    with value.db.session() as session:
        for job_id in terminal_ids[:3]:
            assert session.get(Job, job_id) is None
            assert not session.scalar(select(Attempt).where(Attempt.job_id == job_id))
        assert session.get(Job, terminal_ids[3]).status == "succeeded"
        assert session.get(Job, running).status == "running"
        assert session.get(Job, queued).status == "queued"
        usage = session.scalar(select(InferenceUsage))
        assert usage.used == 2 and usage.reserved == 2
    from fastapi.testclient import TestClient
    from app.hosted import auth
    from app.hosted.main import create_app
    api_settings = replace(value.settings, database_url="postgresql+psycopg://test@localhost/test",
                           r2_endpoint_url="https://test.r2.cloudflarestorage.com")
    app = create_app(api_settings, database=value.db, objects=value.objects)
    token, _ = auth.make_session(value.db, value.uid, 3600)
    with TestClient(app, base_url=api_settings.public_url) as client:
        client.cookies.set(auth.COOKIE, token)
        assert client.get(f"/api/v1/jobs/{terminal_ids[0]}").status_code == 404
        assert client.get(f"/api/v1/jobs/{queued}").status_code == 200


def test_expired_queue_refunds_and_scrubs_payload_without_touching_running_job(setup):
    value = setup
    now = utcnow()
    old = enqueue(value.db, value.settings, value.uid, value.pid, 1, "visual", {"reference_image": "large-private-reference"})["id"]
    active = enqueue(value.db, value.settings, value.uid, value.pid, 1, "text", {"text": "active"})["id"]
    with value.db.session() as session:
        session.get(Job, old).created_at = now - timedelta(hours=25)
        running = session.get(Job, active)
        running.status, running.created_at = "running", now - timedelta(hours=25)
    fresh = enqueue(value.db, value.settings, value.uid, value.pid, 1, "text", {"text": "fresh"})["id"]
    result = reconcile(value.settings, value.db, value.objects, now)
    assert result["expired_queued_jobs"] == 1
    with value.db.session() as session:
        expired = session.get(Job, old)
        assert expired.status == "failed" and expired.error == "queue_expired"
        assert expired.payload == {} and expired.result is None and expired.quota_reserved is False
        assert session.get(Job, active).status == "running"
        assert session.get(Job, fresh).status == "queued"
        usage = session.scalar(select(InferenceUsage))
        assert usage.used == 0 and usage.reserved == 2
    again = reconcile(value.settings, value.db, value.objects, now)
    assert again["expired_queued_jobs"] == 0
    with value.db.session() as session:
        assert session.scalar(select(InferenceUsage)).reserved == 2


def test_chunked_upload_expiry_and_crash_leftovers_are_collected(setup):
    from app.hosted.uploads import upload_lock
    value, now = setup, utcnow()
    expired, active, committed = [str(uuid4()) for _ in range(3)]
    with value.db.session() as session:
        session.get(User, value.uid).reserved_bytes = 30
        session.get(ResourceCounter, 'global').reserved_bytes = 30
        for uid, size, deadline in [(expired, 10, now - timedelta(seconds=1)), (active, 20, now + timedelta(hours=1))]:
            session.add(UploadReservation(id=uid, user_id=value.uid, project_id=value.pid,
                file_name='image.png', bytes=size, expires_at=deadline))
    old = (now - timedelta(hours=25)).replace(tzinfo=timezone.utc).timestamp()
    for uid in (expired, active, committed):
        with upload_lock(value.settings, uid) as path:
            path.write_bytes(b'partial')
            os.utime(path, (old, old))
            os.utime(path.with_suffix('.lock'), (old, old))
    reconcile(value.settings, value.db, value.objects, now)
    directory = value.settings.data_dir / 'uploads'
    assert not (directory / f'{expired}.part').exists()
    assert not (directory / f'{committed}.part').exists()
    assert (directory / f'{active}.part').read_bytes() == b'partial'
    with value.db.session() as session:
        assert session.get(User, value.uid).reserved_bytes == 20
