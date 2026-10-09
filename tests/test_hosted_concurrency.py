"""Real PostgreSQL races; every test uses its own disposable schema.

Set ANNOTATION_TEST_DATABASE_URL to an isolated integration database. These
checks never run against the configured application database implicitly.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
import os
from pathlib import Path
import threading
import uuid

from fastapi import HTTPException
import pytest
from sqlalchemy import create_engine, func, select, text

from app.hosted.config import Settings
from app.hosted.database import Database
from app.hosted import jobs
from app.hosted.models import ImageObject, InferenceUsage, Job, Project, ResourceCounter, User, utcnow
from app.hosted.storage import reserve_upload, release_reservation


@pytest.fixture
def postgres(tmp_path):
    url = os.environ.get("ANNOTATION_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set ANNOTATION_TEST_DATABASE_URL for PostgreSQL concurrency checks")
    url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    schema = "annotation_test_" + uuid.uuid4().hex
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-c search_path={schema}"}, pool_size=20, max_overflow=20)
    settings = Settings(database_url=url, public_url="http://localhost:8765", google_client_id="test",
        google_client_secret="test", r2_endpoint_url="https://test.r2.cloudflarestorage.com",
        r2_access_key_id="test", r2_secret_access_key="test", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache",
        storage_limit_bytes=3000, global_storage_limit_bytes=5000, min_free_disk_bytes=0)
    db = Database(settings, engine)
    db.create_schema()
    with db.session() as session:
        for user_id in ("alice", "bob"):
            session.add(User(id=user_id, google_sub=user_id, email=f"{user_id}@invalid.test", name=user_id))
        session.flush()
        for user_id in ("alice", "bob"):
            session.add(Project(id=user_id, user_id=user_id, name=user_id))
        session.flush()
        for user_id in ("alice", "bob"):
            session.add(ImageObject(id=user_id, project_id=user_id, image_id=1, file_name="image.png",
                object_key=f"{user_id}/image", sha256="0" * 64, size_bytes=0, width=2, height=2, content_type="image/png", status="ready"))
    try:
        yield db, settings
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def race(calls):
    barrier = threading.Barrier(len(calls))
    def invoke(call):
        barrier.wait(timeout=30)
        try:
            return call()
        except HTTPException as exc:
            return exc.status_code
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(invoke, calls))


def test_concurrent_upload_admission_never_overbooks_user_or_global(postgres):
    db, settings = postgres
    responses = race([lambda user=user: reserve_upload(db, settings, user, user, 1000)
                      for user in ("alice", "bob") for _ in range(10)])
    accepted = [value for value in responses if isinstance(value, str)]
    assert len(accepted) == 5
    with db.session() as session:
        assert session.get(ResourceCounter, "global").reserved_bytes == 5000
        assert all(session.get(User, user).reserved_bytes <= 3000 for user in ("alice", "bob"))
        assert sum(session.get(User, user).reserved_bytes for user in ("alice", "bob")) == 5000
    race([lambda id=id: release_reservation(db, id) for id in accepted for _ in range(2)])
    with db.session() as session:
        assert session.get(ResourceCounter, "global").reserved_bytes == 0
        assert all(session.get(User, user).reserved_bytes == 0 for user in ("alice", "bob"))


def test_concurrent_queue_admission_and_repeated_cancel(postgres):
    db, settings = postgres
    responses = race([lambda: jobs.enqueue(db, settings, "alice", "alice", 1, "points", {"revision": 0}) for _ in range(15)])
    accepted = [value for value in responses if isinstance(value, dict)]
    assert len(accepted) == 2
    race([lambda id=job["id"]: jobs.cancel(db, "alice", id) for job in accepted for _ in range(5)])
    with db.session() as session:
        usage = session.get(InferenceUsage, ("alice", utcnow().date().isoformat()))
        assert (usage.used, usage.reserved) == (0, 0)
        assert session.scalar(select(func.count()).select_from(Job).where(Job.status == "cancelled")) == 2


def test_concurrent_daily_quota_and_global_queue(postgres):
    db, settings = postgres
    settings = replace(settings, max_pending_user=50, max_pending_global=5, inference_limit=4)
    with db.session() as session:
        session.add(InferenceUsage(user_id="alice", day=utcnow().date().isoformat(), used=2, reserved=0))
    responses = race([lambda user=user: jobs.enqueue(db, settings, user, user, 1, "points", {})
                      for user in ("alice", "bob") for _ in range(10)])
    assert len([value for value in responses if isinstance(value, dict)]) == 5
    with db.session() as session:
        assert session.scalar(select(func.count()).select_from(Job).where(Job.status == "queued")) == 5
        alice = session.get(InferenceUsage, ("alice", utcnow().date().isoformat()))
        bob = session.get(InferenceUsage, ("bob", utcnow().date().isoformat()))
        assert alice.used + alice.reserved <= 4
        assert bob.used + bob.reserved <= 4
        assert alice.reserved + bob.reserved == 5


def test_parallel_process_startup_is_idempotent(postgres):
    db, settings = postgres
    race([db.create_schema for _ in range(12)])
    with db.session() as session:
        assert session.scalar(select(func.count()).select_from(ResourceCounter)) == 1


def test_concurrent_metadata_generations_respect_account_and_global_quotas(postgres):
    from app.hosted.quota import logical_metadata_bytes, mutate_project, recover_project
    from app.hosted.storage import create_project_store, FilesystemStore
    from app.hosted.models import MetadataMutation
    db, settings = postgres
    settings = replace(settings, storage_limit_bytes=2000, global_storage_limit_bytes=3000)
    objects = FilesystemStore(settings.data_dir / "objects")
    for user in ("alice", "bob"):
        store = create_project_store(settings, user, user)
        size = logical_metadata_bytes(store.database)
        with db.session() as session:
            project = session.get(Project, user)
            project.quota_version, project.metadata_bytes = 1, size
            session.get(User, user).storage_bytes = size
            session.get(ResourceCounter, "global").storage_bytes += size
    responses = race([lambda user=user, index=index: mutate_project(settings, db, objects, user, user,
        lambda store, *_: store.add_category(f"{index:02}-" + "class" * 20, "#112233"))
        for user in ("alice", "bob") for index in range(12)])
    accepted = [result for result in responses if isinstance(result, dict)]
    assert 2 < len(accepted) < 24
    for user in ("alice", "bob"):
        recover_project(settings, db, user, objects)
    with db.session() as session:
        total = 0
        for user in ("alice", "bob"):
            actual = logical_metadata_bytes(settings.data_dir / "projects" / user / "proyecto.sqlite3")
            account = session.get(User, user)
            assert account.storage_bytes == actual <= settings.storage_limit_bytes
            assert account.reserved_bytes == 0
            assert session.get(Project, user).metadata_bytes == actual
            total += actual
        assert session.get(ResourceCounter, "global").storage_bytes == total <= settings.global_storage_limit_bytes
        assert session.get(ResourceCounter, "global").reserved_bytes == 0
        assert list(session.scalars(select(MetadataMutation))) == []


@pytest.fixture(params=['sqlite', 'postgres'])
def export_hosted(request, tmp_path):
    """The same bounded race runs locally and on real PostgreSQL in CI."""
    from fastapi.testclient import TestClient
    from sqlalchemy import delete
    from app.hosted import auth
    from app.hosted.main import create_app
    from app.hosted.storage import FilesystemStore

    if request.param == 'postgres':
        db, settings = request.getfixturevalue('postgres')
        with db.session() as session:
            session.execute(delete(ImageObject))
            session.execute(delete(Project))
    else:
        settings = Settings(database_url='postgresql+psycopg://unused/test', public_url='http://localhost:8765',
            google_client_id='test', google_client_secret='test', r2_endpoint_url='https://test.r2.cloudflarestorage.com',
            r2_access_key_id='test', r2_secret_access_key='test', data_dir=tmp_path/'data', cache_dir=tmp_path/'cache',
            min_free_disk_bytes=0)
        db = Database(settings, create_engine(f'sqlite:///{tmp_path / "metadata.sqlite3"}'))
        db.create_schema()
        with db.session() as session:
            for user in ('alice', 'bob'):
                session.add(User(id=user, google_sub=user, email=f'{user}@example.invalid', name=user))
    settings = replace(settings, storage_limit_bytes=1_000_000, global_storage_limit_bytes=2_000_000)
    objects = FilesystemStore(tmp_path/'objects')
    app = create_app(settings, database=db, objects=objects)
    clients = {}
    for user in ('alice', 'bob'):
        token, row = auth.make_session(db, user, settings.session_seconds)
        client = TestClient(app, base_url=settings.public_url, raise_server_exceptions=False)
        client.cookies.set(auth.COOKIE, token)
        client.headers.update({'X-CSRF-Token': row.csrf_token, 'X-Annotation-State-Version': '2'})
        clients[user] = client
    try:
        yield settings, db, objects, clients
    finally:
        for client in clients.values():
            client.close()
        if request.param == 'sqlite':
            db.engine.dispose()


@pytest.mark.parametrize('phase', ['preview', 'generate', 'image_fetch', 'upload', 'download'])
def test_yolo_releases_global_lock_and_freezes_authorized_generation(export_hosted, monkeypatch, phase):
    import io
    import zipfile
    from app import yolo
    from app.hosted.storage import ObjectCache
    from test_hosted_backend import project_and_image
    from test_hosted_quota import assert_accounted

    settings, db, objects, clients = export_hosted
    alice, bob = clients['alice'], clients['bob']
    project = project_and_image(alice)
    root = f"/api/v1/projects/{project['id']}"
    bob_project = project_and_image(bob)
    category = alice.post(root+'/categories', json={'name': 'Box'}).json()['id']
    state = alice.get(root+'/images/1/state').json()
    state.update(schema_version=2, annotations=[{'id':'box','kind':'bbox','category_id':category,'iscrowd':0,'bbox':[0,0,8,6]}])
    state = alice.put(root+'/images/1/state', json=state).json()
    options = {'task':'detection','include_images':True}
    preview = alice.post(root+'/yolo/preview', json=options).json()
    payload = {**options,'snapshot':preview['snapshot']}
    if phase == 'download':
        export = alice.post(root+'/yolo/export', json=payload).json()
        download = root+'/yolo/download/'+export['export_id']
    entered, release = threading.Event(), threading.Event()
    target, method = {'preview': (yolo,'prepare'), 'generate': (yolo,'generate'),
        'image_fetch': (ObjectCache,'get'), 'upload': (objects,'put'), 'download': (objects,'open_stream')}[phase]
    original = getattr(target, method)
    def stalled(*args, **kwargs):
        entered.set()
        assert release.wait(20), 'test did not release slow export work'
        return original(*args, **kwargs)
    monkeypatch.setattr(target, method, stalled)
    def unrelated_work():
        # Account/global admission and job admission must complete while the
        # first export is still waiting, including on PostgreSQL row locks.
        reservation = reserve_upload(db, settings, 'bob', bob_project['id'], 100)
        release_reservation(db, reservation)
        job = jobs.enqueue(db, settings, 'bob', bob_project['id'], 1, 'points', {})
        jobs.cancel(db, 'bob', job['id'])
        state['annotations'][0]['bbox'] = [4,3,4,3]
        changed = alice.put(root+'/images/1/state', json=state)
        assert changed.status_code == 200, changed.text
        assert_accounted(settings, db)
    with ThreadPoolExecutor(2) as pool:
        export = pool.submit(lambda: alice.get(download) if phase == 'download' else
            alice.post(root+('/yolo/preview' if phase == 'preview' else '/yolo/export'),
                       json=options if phase == 'preview' else payload))
        try:
            assert entered.wait(10)
            pool.submit(unrelated_work).result(timeout=5)
        finally:
            release.set()
        response = export.result(timeout=20)
    assert response.status_code == 200, response.text
    if phase == 'preview':
        assert response.json()['snapshot'] == preview['snapshot']
    else:
        if phase != 'download':
            response = alice.get(root+'/yolo/download/'+response.json()['export_id'])
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            # The concurrent edit cannot leak into this generation.
            assert archive.read('labels/folder/sample.txt') == b'0 0.25 0.25 0.5 0.5\n'
    assert alice.post(root+'/yolo/export', json=payload).status_code == 409
    assert_accounted(settings, db)
    assert not list(settings.data_dir.glob('coco-*'))


def test_export_revoked_during_upload_is_discarded_without_quota_changes(export_hosted, monkeypatch):
    from test_hosted_backend import project_and_image
    from test_hosted_quota import assert_accounted
    settings, db, objects, clients = export_hosted
    alice = clients['alice']
    project = project_and_image(alice)
    root = f"/api/v1/projects/{project['id']}"
    options = {'task':'detection', 'include_empty':True}
    preview = alice.post(root+'/yolo/preview', json=options).json()
    put = objects.put
    def revoke(key, data, content_type=None):
        put(key, data, content_type)
        assert alice.delete(root).status_code == 200
    monkeypatch.setattr(objects, 'put', revoke)
    response = alice.post(root+'/yolo/export', json={**options,'snapshot':preview['snapshot']})
    assert response.status_code == 404, response.text
    assert not list(objects.list('exports/'))
    assert not list(settings.data_dir.glob('coco-*'))
    assert_accounted(settings, db)


def test_concurrent_account_picture_migration_preserves_legacy_readers(postgres):
    db, _ = postgres
    with db.engine.begin() as connection:
        connection.execute(text("ALTER TABLE users DROP COLUMN picture"))
    # Independent web processes can start against the same pre-upgrade schema.
    assert race([db.create_schema for _ in range(4)]) == [None] * 4
    with db.session() as session:
        users = list(session.scalars(select(User).order_by(User.id)))
        assert [user.id for user in users] == ["alice", "bob"]
        assert all(user.picture is None and user.storage_bytes == 0 and user.reserved_bytes == 0 for user in users)
        users[0].picture = "https://lh3.googleusercontent.com/a/profile"
    # Old software continues selecting/updating only the fields it understands.
    with db.engine.begin() as connection:
        connection.execute(text("UPDATE users SET name='Updated' WHERE id='alice'"))
        assert connection.execute(text("SELECT email FROM users WHERE id='alice'")).scalar_one() == "alice@invalid.test"
    with db.session() as session:
        assert session.get(User, "alice").picture == "https://lh3.googleusercontent.com/a/profile"
