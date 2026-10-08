"""Multi-user API tests with explicitly injected, disposable storage and metadata."""
from dataclasses import replace
from datetime import timedelta
import io
import json
from urllib.parse import parse_qs, urlsplit

from authlib.jose import JsonWebKey, JsonWebToken
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from app.hosted import auth
from app.hosted.config import Settings
from app.hosted.database import Database
from app.hosted.main import create_app
from app.hosted.models import AuthFlow, AuthSession, ImageObject, InferenceUsage, Job, MetadataMutation, Project, ResourceCounter, UploadReservation, User, utcnow
from app.hosted.storage import FilesystemStore, ObjectCache


def image_bytes(size=(16, 12), color="red"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def hosted(tmp_path):
    settings = Settings(database_url="postgresql+psycopg://unused/test", public_url="http://localhost:8765",
        google_client_id="test-client", google_client_secret="test-secret", r2_endpoint_url="https://test.r2.cloudflarestorage.com",
        r2_access_key_id="test-key", r2_secret_access_key="test-secret", data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache", min_free_disk_bytes=0)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db = Database(settings, engine=engine)
    objects = FilesystemStore(tmp_path / "objects")
    app = create_app(settings, database=db, objects=objects)
    with db.session() as session:
        for id in ("alice", "bob"):
            session.add(User(id=id, google_sub=f"google-{id}", email=f"{id}@example.invalid", name=id))
    clients = {}
    for id in ("alice", "bob"):
        token, row = auth.make_session(db, id, settings.session_seconds)
        client = TestClient(app, base_url=settings.public_url, raise_server_exceptions=False)
        client.cookies.set(auth.COOKIE, token)
        client.headers["X-CSRF-Token"] = row.csrf_token
        clients[id] = client
    yield settings, db, objects, app, clients
    for client in clients.values():
        client.close()
    engine.dispose()


def project_and_image(client):
    response = client.post("/api/v1/projects", json={"name": "Demo"})
    assert response.status_code == 201, response.text
    project = response.json()
    response = client.post(f"/api/v1/projects/{project['id']}/images", files={"file": ("sample.png", image_bytes(), "image/png")},
                           data={"relative_path": "folder/sample.png"})
    assert response.status_code == 201, response.text
    return response.json()


def test_hosted_state_and_inference_skip_unused_mask_previews(hosted, monkeypatch):
    import numpy as np
    from app.geometry import mask_payload

    *_, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    path = f"/api/v1/projects/{project['id']}"
    category = client.post(path + "/categories", json={"name": "Object", "color": "#33aa99"}).json()
    state = client.get(path + "/images/1/state").json()
    annotation = {"id": "saved-object", "category_id": category["id"], "iscrowd": 0,
                  **mask_payload(np.ones((12, 16), dtype=bool))}
    state["annotations"] = [annotation]

    def unexpected(*args, **kwargs):
        pytest.fail("The hosted editor does not use PNG mask previews")

    monkeypatch.setattr("app.storage.mask_preview", unexpected)
    response = client.put(path + "/images/1/state", json=state)
    assert response.status_code == 200, response.text
    saved = response.json()
    assert "preview" not in saved["annotations"][0]
    for key in ("mask", "components", "controls"):
        assert saved["annotations"][0][key] == annotation[key]
    assert client.get(path + "/images/1/state").json() == saved
    response = client.post(path + "/infer/points", json={"image_id": 1, "revision": saved["revision"],
                           "part": {"points": [{"x": 5, "y": 5, "label": 1}]}})
    assert response.status_code == 202, response.text
    assert clients["bob"].get(path + "/images/1/state").status_code == 404


@pytest.mark.parametrize("outcome", ["succeeded", "cancelled", "deleted"])
def test_job_wait_delivers_completion_and_rechecks_ownership(hosted, monkeypatch, outcome):
    _, db, _, _, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    job = client.post(f"/api/v1/projects/{project['id']}/infer/points", json={
        "image_id": 1, "revision": 0, "part": {"points": [{"x": 5, "y": 5, "label": 1}]}}).json()
    path = f"/api/v1/jobs/{job['id']}"
    assert client.get(path).json()["status"] == "queued"
    assert client.get(path + "?wait_ms=10").json()["status"] == "queued"
    assert client.get(path + "?wait_ms=1001").status_code == 422
    assert clients["bob"].get(path + "?wait_ms=1000").status_code == 404

    async def complete_after_read(seconds):
        # The read transaction must have ended before waiting. Commit a new
        # state as an independent dispatcher/deletion request would do.
        with db.session() as session:
            if outcome == "deleted":
                session.get(Project, project["id"]).deleted_at = utcnow()
            else:
                row = session.get(Job, job["id"])
                row.status = outcome
                row.result = {"image_id": 1, "revision": 0}

    monkeypatch.setattr("app.hosted.main.asyncio.sleep", complete_after_read)
    response = client.get(path + "?wait_ms=1000")
    if outcome == "deleted":
        assert response.status_code == 404
    else:
        assert response.status_code == 200
        assert response.json()["status"] == outcome
        if outcome == "succeeded":
            assert response.json()["result"] == {"image_id": 1, "revision": 0}
    assert response.headers["cache-control"] == "private, no-store"


def test_hosted_config_fails_closed(hosted):
    settings, *_ = hosted
    for bad in [replace(settings, environment="prod"), replace(settings, database_url="sqlite://"),
                replace(settings, public_url="http://example.com"), replace(settings, r2_bucket="other"),
                replace(settings, r2_endpoint_url="http://test.r2.cloudflarestorage.com"),
                replace(settings, public_url="https://example.com/callback"), replace(settings, primary_worker_url="http://worker")]:
        with pytest.raises(ValueError):
            bad.validate()
    replace(settings, environment="prod", r2_bucket="annotation-prod", public_url="https://annotation.example.com").validate()


def test_isolation_private_images_state_export_and_no_paths(hosted):
    settings, db, objects, app, clients = hosted
    alice, bob = clients["alice"], clients["bob"]
    project = project_and_image(alice)
    root = f"/api/v1/projects/{project['id']}"
    assert "directory" not in project and "image_root" not in project
    assert bob.get("/api/v1/projects").json() == {"projects": []}
    for suffix in ("", "/images/1/file", "/images/1/state", "/coco/download"):
        assert bob.get(root + suffix).status_code == 404
    for suffix in ("/categories", "/coco/export"):
        assert bob.post(root + suffix, json={"name": "Other"}).status_code == 404
    assert alice.get(root + "/images/1/file").headers["cache-control"] == "private, no-store"
    assert alice.get(root + "/images/1/file?thumbnail=true").status_code == 200
    assert alice.get("/api/browse").status_code == 404
    assert alice.get("/api/project").status_code == 404
    assert alice.get("/privacy").status_code == 200
    assert alice.post(root + "/coco/export").json() == {"file_name": "anotaciones.coco.json"}
    coco = alice.get(root + "/coco/download").json()
    assert coco["images"][0]["file_name"] == "folder/sample.png"
    assert str(settings.data_dir) not in json.dumps(coco)
    with db.session() as session:
        image = session.scalar(select(ImageObject).where(ImageObject.project_id == project["id"]))
        metadata = session.get(Project, project["id"]).metadata_bytes
        assert session.get(User, "alice").storage_bytes == len(image_bytes()) + image.thumbnail_bytes + metadata
        assert session.get(User, "alice").reserved_bytes == 0
        assert list(session.scalars(select(UploadReservation))) == []


def test_sessions_csrf_origin_and_expiry(hosted):
    settings, db, objects, app, clients = hosted
    anonymous = TestClient(app, base_url=settings.public_url)
    assert anonymous.get("/api/v1/auth/session").json()["user"] is None
    assert anonymous.post("/api/v1/projects", json={"name": "No"}).status_code == 401
    client = clients["alice"]
    assert client.get("/api/v1/auth/session").json()["user"]["id"] == "alice"
    assert client.post("/api/v1/projects", json={"name": "No"}, headers={"X-CSRF-Token": "bad"}).status_code == 403
    assert client.post("/api/v1/projects", json={"name": "No"}, headers={"Origin": "https://other.invalid"}).status_code == 403
    with db.session() as session:
        row = session.get(AuthSession, auth.digest(client.cookies.get(auth.COOKIE)))
        row.expires_at = utcnow() - timedelta(seconds=1)
    assert client.get("/api/v1/auth/session").json()["user"] is None


def test_upload_validation_quota_and_r2_failure_cleanup(hosted, monkeypatch):
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project_id = client.post("/api/v1/projects", json={"name": "Uploads"}).json()["id"]
    endpoint = f"/api/v1/projects/{project_id}/images"
    for filename, contents in [("../bad.png", image_bytes()), ("bad.png", b"not an image"), ("bad.txt", image_bytes())]:
        assert client.post(endpoint, files={"file": (filename, contents)}).status_code == 422
    monkeypatch.setattr(objects, "put", lambda *a, **kw: (_ for _ in ()).throw(OSError("private-endpoint should not leak")))
    response = client.post(endpoint, files={"file": ("sample.png", image_bytes())})
    assert response.status_code == 503
    assert "private-endpoint" not in response.text
    with db.session() as session:
        assert list(session.scalars(select(ImageObject))) == []
        assert session.get(User, "alice").storage_bytes == session.get(Project, project_id).metadata_bytes
        assert session.get(User, "alice").reserved_bytes == 0
    with db.session() as session:
        session.get(User, "alice").storage_bytes = settings.storage_limit_bytes
    assert client.post(endpoint, files={"file": ("sample.png", image_bytes())}).status_code == 413


def test_mask_save_revision_and_coco_roundtrip(hosted):
    _, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    category = client.post(root + "/categories", json={"name": "Object"}).json()
    components = [{"outer": [[1, 1], [5, 1], [5, 5], [1, 5]], "holes": []}]
    geometry = client.post(root + "/geometry", json={"image_id": 1, "components": components})
    assert geometry.status_code == 200, geometry.text
    state = client.get(root + "/images/1/state").json()
    state["annotations"] = [{"id": "object-1", "category_id": category["id"], "iscrowd": 0, **geometry.json()}]
    saved = client.put(root + "/images/1/state", json=state)
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == 1
    assert client.put(root + "/images/1/state", json=state).status_code == 409
    client.post(root + "/coco/export")
    source = client.get(root + "/coco/download").content
    fresh = project_and_image(client)
    response = client.post(f"/api/v1/projects/{fresh['id']}/coco/import", files={"file": ("annotations.json", source, "application/json")})
    assert response.status_code == 200, response.text
    imported = client.get(f"/api/v1/projects/{fresh['id']}/images/1/state").json()
    assert imported["annotations"][0]["mask"] == saved.json()["annotations"][0]["mask"]
    assert client.post(root + "/coco/import", files={"file": ("annotations.json", source)}).status_code == 409


def test_queue_quota_cancel_and_deletion(hosted):
    _, db, objects, app, clients = hosted
    client, other = clients["alice"], clients["bob"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    body = {"image_id": 1, "revision": 0, "part": {"points": [{"x": 3, "y": 3, "label": 1}]}}
    jobs = [client.post(root + "/infer/points", json=body) for _ in range(3)]
    assert [r.status_code for r in jobs] == [202, 202, 429]
    id = jobs[0].json()["id"]
    assert other.get(f"/api/v1/jobs/{id}").status_code == 404
    assert other.post(f"/api/v1/jobs/{id}/cancel").status_code == 404
    assert client.post(f"/api/v1/jobs/{id}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/v1/jobs/{id}/cancel").json()["status"] == "cancelled"
    with db.session() as session:
        usage = session.get(InferenceUsage, ("alice", utcnow().date().isoformat()))
        assert (usage.used, usage.reserved) == (0, 1)
    assert client.delete(root).status_code == 200
    assert client.get(root + "/images/1/file").status_code == 404
    assert client.get(f"/api/v1/jobs/{id}").status_code == 404
    with db.session() as session:
        assert session.get(User, "alice").storage_bytes == 0
        assert session.get(InferenceUsage, ("alice", utcnow().date().isoformat())).reserved == 0
        record = session.scalar(select(ImageObject).where(ImageObject.project_id == project["id"]))
        assert record.status == "deleted" and objects.get(record.object_key)


def test_cache_recovery_checksum_and_capacity(tmp_path):
    objects = FilesystemStore(tmp_path / "objects")
    cache = ObjectCache(tmp_path / "cache", max_bytes=5)
    for key, data in [("a", b"123"), ("b", b"456")]:
        import hashlib
        digest = hashlib.sha256(data).hexdigest()
        objects.put(key, data)
        assert cache.get(objects, key, digest) == data
    assert sum(p.stat().st_size for p in cache.root.iterdir() if not p.name.startswith(".")) <= 5
    with pytest.raises(ValueError, match="checksum"):
        cache.get(objects, "a", "0" * 64)


@pytest.mark.parametrize("wrong_claim", [None, "nonce", "aud", "iss", "exp"])
def test_google_callback_signature_claims_state_and_replay(hosted, monkeypatch, wrong_claim):
    settings, db, objects, app, clients = hosted
    client = TestClient(app, base_url=settings.public_url)
    login = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert login.status_code == 302
    params = parse_qs(urlsplit(login.headers["location"]).query)
    assert set(params["scope"][0].split()) == {"openid", "email", "profile"}
    assert params["code_challenge_method"] == ["S256"]
    now = int(utcnow().timestamp())
    # utcnow is naive UTC; timestamp is host-local, so use aware clock for JWT.
    from datetime import datetime, timezone
    now = int(datetime.now(timezone.utc).timestamp())
    claims = {"iss": "https://accounts.google.com", "aud": settings.google_client_id, "sub": "google-new",
        "iat": now, "exp": now + 3600, "nonce": params["nonce"][0], "email": "verified@example.invalid", "email_verified": True}
    if wrong_claim:
        claims[wrong_claim] = now - 100 if wrong_claim == "exp" else "wrong"
    key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
    token = JsonWebToken(["RS256"]).encode({"alg": "RS256", "kid": "test"}, claims, key).decode()
    public = key.as_dict(is_private=False)
    public["kid"] = "test"

    class OAuthClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def fetch_token(self, *a, **kw): return {"id_token": token, "access_token": "test"}

    class HTTPClient(OAuthClient):
        async def get(self, *a, **kw):
            import httpx
            return httpx.Response(200, json={"keys": [public]}, request=httpx.Request("GET", "https://www.googleapis.com/oauth2/v3/certs"))

    monkeypatch.setattr(auth, "AsyncOAuth2Client", OAuthClient)
    monkeypatch.setattr(auth.httpx, "AsyncClient", HTTPClient)
    callback = f"/api/v1/auth/google/callback?state={params['state'][0]}&code=example"
    response = client.get(callback, follow_redirects=False)
    assert response.status_code == (400 if wrong_claim else 302), response.text
    if not wrong_claim:
        assert "HttpOnly" in response.headers["set-cookie"]
        assert client.get("/api/v1/auth/session").json()["user"]["email"] == "verified@example.invalid"
    assert client.get(callback, follow_redirects=False).status_code == 400


def test_google_denial_invalid_state_and_logout(hosted):
    settings, db, objects, app, clients = hosted
    client = TestClient(app, base_url=settings.public_url)
    login = client.get("/api/v1/auth/google/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    assert client.get("/api/v1/auth/google/callback?state=invalid&code=x").status_code == 400
    denied = client.get(f"/api/v1/auth/google/callback?state={state}&error=access_denied", follow_redirects=False)
    assert denied.status_code == 302 and "error=cancelled" in denied.headers["location"]
    assert clients["alice"].post("/api/v1/auth/logout").status_code == 200
    assert clients["alice"].get("/api/v1/auth/session").json()["user"] is None


def test_forged_mask_size_and_aggregate_budget_rejected_before_decoding(hosted, monkeypatch):
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    import app.hosted.main as module
    monkeypatch.setattr(module, "decode_mask", lambda *a: pytest.fail("Untrusted dimensions reached native decoder"))
    response = client.post(root + "/masks/fill-holes", json={"image_id": 1, "mask": {"size": [15000, 10000], "counts": [150000000]}})
    assert response.status_code == 422
    limited = create_app(replace(settings, max_mask_pixels_per_request=192), database=db, objects=objects)
    limited_client = TestClient(limited, base_url=settings.public_url)
    limited_client.cookies.update(client.cookies)
    limited_client.headers["X-CSRF-Token"] = client.headers["X-CSRF-Token"]
    monkeypatch.setattr(module, "union_masks", lambda *a: pytest.fail("Over-budget masks reached decoder"))
    response = limited_client.post(root + "/masks/union", json={"image_id": 1,
        "masks": [{"size": [12, 16], "counts": [0, 192]}] * 2})
    assert response.status_code == 413


def test_ambiguous_commit_keeps_successfully_registered_objects(hosted, monkeypatch):
    from contextlib import contextmanager
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    pid = client.post("/api/v1/projects", json={"name": "Ambiguous commit"}).json()["id"]
    original_session, raised = db.session, False

    @contextmanager
    def ambiguous_commit():
        nonlocal raised
        committed_ready = False
        with original_session() as session:
            yield session
            committed_ready = any(isinstance(value, MetadataMutation) and value.image_object_id for value in session.deleted)
        if committed_ready and not raised:
            raised = True
            raise OSError("Connection lost after committing")

    monkeypatch.setattr(db, "session", ambiguous_commit)
    response = client.post(f"/api/v1/projects/{pid}/images", files={"file": ("sample.png", image_bytes())})
    assert response.status_code == 503 and raised
    with db.session() as session:
        row = session.scalar(select(ImageObject).where(ImageObject.project_id == pid))
        assert row.status == "ready"
        assert objects.get(row.object_key) == image_bytes()
        assert session.get(User, "alice").storage_bytes == len(image_bytes()) + row.thumbnail_bytes + session.get(Project, pid).metadata_bytes
    assert client.get(f"/api/v1/projects/{pid}").json()["images"][0]["id"] == 1
    assert client.get(f"/api/v1/projects/{pid}/images/1/state").status_code == 200


def test_job_storage_guard_and_prompt_scrub(hosted, monkeypatch):
    from app.hosted import jobs as job_service
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    payload = {"image_id": 1, "revision": 0, "part": {"points": [{"x": 1, "y": 1, "label": 1}]}}
    result = client.post(root + "/infer/points", json=payload).json()
    client.post(f"/api/v1/jobs/{result['id']}/cancel")
    with db.session() as session:
        assert session.get(Job, result["id"]).payload == {}
    import shutil
    monkeypatch.setattr(job_service.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100, 100, 0))
    with pytest.raises(Exception) as error:
        job_service.enqueue(db, replace(settings, min_free_disk_bytes=1), "alice", project["id"], 1, "points", payload)
    assert getattr(error.value, "status_code", None) == 503


def test_coco_export_excludes_pending_sqlite_residue(hosted):
    from app.hosted.models import identity
    from app.hosted.storage import project_store, register_image
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    category = client.post(root + "/categories", json={"name": "Object"}).json()
    orphan = ImageObject(id=identity(), project_id=project["id"], image_id=2, file_name="orphan.png",
        object_key=f"projects/{project['id']}/orphan", sha256="0" * 64, size_bytes=100, width=16, height=12,
        content_type="image/png", status="pending")
    with db.session() as session:
        session.add(orphan)
    store = project_store(settings, project["id"])
    register_image(store, orphan)
    state = store.get_state(2)
    state["annotations"] = [{"id": "orphan-annotation", "category_id": category["id"],
                             "mask": {"size": [12, 16], "counts": [0, 192]}}]
    store.save_state(2, state)
    assert len(store.project()["images"]) == 2
    assert client.post(root + "/coco/export").status_code == 200
    exported = client.get(root + "/coco/download").json()
    assert [image["id"] for image in exported["images"]] == [1]
    assert exported["annotations"] == []


def test_job_status_returns_missing_when_retention_removes_project_between_reads(hosted, monkeypatch):
    from sqlalchemy.orm import Session
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    job = client.post(f"/api/v1/projects/{project['id']}/infer/points", json={
        "image_id": 1, "revision": 0, "part": {"points": [{"x": 1, "y": 1, "label": 1}]}}).json()
    original_get = Session.get

    def project_disappears(session, entity, key, *args, **kwargs):
        if entity is Project and key == project["id"]:
            return None
        return original_get(session, entity, key, *args, **kwargs)

    monkeypatch.setattr(Session, "get", project_disappears)
    assert client.get(f"/api/v1/jobs/{job['id']}").status_code == 404


def test_large_chunked_image_keeps_original_and_quota(hosted):
    """Both old limits are exceeded; transfer chunks never exceed the proxy cap."""
    settings, db, objects, app, clients = hosted
    client = clients['alice']
    pid = client.post('/api/v1/projects', json={'name': 'Large original'}).json()['id']
    root = f'/api/v1/projects/{pid}'
    output = io.BytesIO()
    Image.new('RGB', (4097, 4096), 'red').save(output, format='BMP')
    data = output.getvalue()
    assert len(data) > 20 * 1024 * 1024
    response = client.post(root + '/uploads', json={'file_name': 'survey/large.bmp', 'size': len(data)})
    assert response.status_code == 201, response.text
    upload = response.json()
    endpoint = root + '/uploads/' + upload['id']
    with db.session() as session:
        assert session.get(User, 'alice').reserved_bytes == len(data)
    assert clients['bob'].put(endpoint + '?offset=0', content=b'x').status_code == 404
    assert client.post(endpoint + '/complete').status_code == 409
    assert client.put(endpoint + '?offset=1', content=b'x').status_code == 409
    for offset in range(0, len(data), upload['chunk_bytes']):
        chunk = data[offset:offset + upload['chunk_bytes']]
        response = client.put(endpoint + f'?offset={offset}', content=chunk)
        assert response.status_code == 200, response.text
        assert response.json()['offset'] == offset + len(chunk)
        if offset == 0:
            assert client.put(endpoint + '?offset=0', content=chunk).json() == response.json()
            assert client.put(endpoint + '?offset=0', content=b'wrong').status_code == 409
    response = client.post(endpoint + '/complete')
    assert response.status_code == 201, response.text
    assert response.json()['images'][0]['width'] == 4097
    assert response.json()['images'][0]['height'] == 4096
    assert client.post(endpoint + '/complete').json() == response.json()
    assert client.post(root + '/coco/export').status_code == 200
    coco = client.get(root + '/coco/download').json()
    assert coco['images'][0]['width'] == 4097
    assert coco['images'][0]['file_name'] == 'survey/large.bmp'
    with db.session() as session:
        row = session.get(ImageObject, upload['id'])
        assert objects.get(row.object_key) == data
        assert session.get(User, 'alice').reserved_bytes == 0
        assert session.get(ResourceCounter, 'global').reserved_bytes == 0
    assert not (settings.data_dir / 'uploads' / (upload['id'] + '.part')).exists()


def test_chunked_upload_enforces_reserved_bytes_and_releases_cancellation(hosted):
    settings, db, objects, app, clients = hosted
    client = clients['alice']
    pid = client.post('/api/v1/projects', json={'name': 'Interrupted'}).json()['id']
    root = f'/api/v1/projects/{pid}/uploads'
    assert client.post(root, json={'file_name': '../private.png', 'size': 50}).status_code == 422
    assert client.post(root, json={'file_name': 'a.png', 'size': settings.storage_limit_bytes}).status_code == 413
    uid = client.post(root, json={'file_name': 'a.png', 'size': 5}).json()['id']
    endpoint = root + '/' + uid
    assert client.put(endpoint + '?offset=0', content=b'123456').status_code == 409
    assert client.put(endpoint + '?offset=0', content=b'123').status_code == 200
    assert client.put(endpoint + '?offset=3', content=b'45').status_code == 200
    assert client.post(endpoint + '/complete').status_code == 422
    assert clients['bob'].delete(endpoint).status_code == 404
    assert client.delete(endpoint).status_code == 200
    assert client.put(endpoint + '?offset=0', content=b'123').status_code == 404
    with db.session() as session:
        assert session.get(User, 'alice').reserved_bytes == 0
    assert not (settings.data_dir / 'uploads' / (uid + '.part')).exists()


def test_native_image_response_preserves_original_bytes_and_exif_pixel_grid(hosted):
    _, _, _, _, clients = hosted
    client = clients['alice']
    pid = client.post('/api/v1/projects', json={'name': 'Native originals'}).json()['id']
    root = f'/api/v1/projects/{pid}'
    source = Image.new('RGB', (120, 80), 'red')
    for n, orientation in enumerate((1, 6), 1):
        exif = Image.Exif(); exif[274] = orientation
        data = io.BytesIO(); source.save(data, 'JPEG', exif=exif)
        raw = data.getvalue()
        assert client.post(root + '/images', files={'file': (f'original-{n}.jpg', raw)}).status_code == 201
        result = client.get(root + f'/images/{n}/file')
        assert result.headers['cache-control'] == 'private, no-store'
        with Image.open(io.BytesIO(result.content)) as display:
            assert display.size == (120, 80)
            assert display.getexif().get(274, 1) == 1
            with Image.open(io.BytesIO(raw)) as original:
                assert display.convert('RGB').tobytes() == original.convert('RGB').tobytes()
        if orientation == 1:
            assert result.content == raw
            assert result.headers['content-type'] == 'image/jpeg'
        else:
            assert result.headers['content-type'] == 'image/png'
        assert clients['bob'].get(root + f'/images/{n}/file').status_code == 404
