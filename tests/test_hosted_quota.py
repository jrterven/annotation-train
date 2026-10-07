"""Logical data quotas and failures across PostgreSQL/SQLite commit boundaries."""
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
import pytest

from test_hosted_backend import hosted, project_and_image, image_bytes
from app.hosted.main import create_app
from app.hosted.models import ImageObject, MetadataMutation, Project, ResourceCounter, UploadReservation, User
from app.hosted import quota


def assert_accounted(settings, db):
    with db.session() as session:
        users = list(session.scalars(select(User)))
        for user in users:
            expected = 0
            for project in session.scalars(select(Project).where(Project.user_id == user.id, Project.deleted_at.is_(None))):
                actual = quota.logical_metadata_bytes(settings.data_dir / "projects" / project.id / "proyecto.sqlite3")
                assert project.metadata_bytes == actual
                assert project.pending_mutation_id is None
                expected += actual
                expected += sum(image.size_bytes + image.thumbnail_bytes for image in session.scalars(select(ImageObject).where(
                    ImageObject.project_id == project.id, ImageObject.status == "ready")))
            assert user.storage_bytes == expected
            assert user.reserved_bytes == 0
        assert session.get(ResourceCounter, "global").storage_bytes == sum(u.storage_bytes for u in users)
        assert session.get(ResourceCounter, "global").reserved_bytes == 0
        assert list(session.scalars(select(MetadataMutation))) == []


def test_all_logical_bytes_count_on_create_update_import_and_delete(hosted):
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    assert_accounted(settings, db)
    category = client.post(root + "/categories", json={"name": "X" * 120}).json()
    with db.session() as session:
        large = session.get(User, "alice").storage_bytes
    assert client.patch(root + f"/categories/{category['id']}", json={"name": "X"}).status_code == 200
    with db.session() as session:
        assert session.get(User, "alice").storage_bytes == large - 119
    state = client.get(root + "/images/1/state").json()
    state["annotations"] = [{"id": "instance", "category_id": category["id"], "mask": {"size": [12, 16], "counts": [0, 192]}}]
    assert client.put(root + "/images/1/state", json=state).status_code == 200
    assert_accounted(settings, db)
    client.post(root + "/coco/export")
    exported = client.get(root + "/coco/download").content
    fresh = project_and_image(client)
    target = f"/api/v1/projects/{fresh['id']}"
    assert client.post(target + "/coco/import", files={"file": ("data.json", exported)}).status_code == 200
    assert_accounted(settings, db)
    client.delete(root)
    client.delete(target)
    assert_accounted(settings, db)
    # Retained backup originals remain physical objects, but no longer active quota.
    assert len(list(objects.list("projects/"))) == 4
    with db.session() as session:
        assert session.get(User, "alice").storage_bytes == 0


def test_metadata_and_thumbnail_quota_reject_before_live_generation_changes(hosted):
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    first = project_and_image(client)
    with db.session() as session:
        first_cost = session.get(Project, first["id"]).metadata_bytes
        used = session.get(User, "alice").storage_bytes
    empty = client.post("/api/v1/projects", json={"name": "Demo"}).json()
    with db.session() as session:
        old = session.get(Project, empty["id"]).metadata_bytes
        current_used = session.get(User, "alice").storage_bytes
        image = session.scalar(select(ImageObject).where(ImageObject.project_id == first["id"]))
        required_delta = first_cost - old + image.size_bytes + image.thumbnail_bytes
    # The multipart upload itself fits. Thumbnails plus canonical metadata push
    # the finalized content over quota, so neither an image nor a partial state survives.
    limited_settings = replace(settings, storage_limit_bytes=current_used + required_delta - 1)
    limited = TestClient(create_app(limited_settings, database=db, objects=objects), base_url=settings.public_url)
    limited.cookies.update(client.cookies)
    limited.headers["X-CSRF-Token"] = client.headers["X-CSRF-Token"]
    root = f"/api/v1/projects/{empty['id']}"
    response = limited.post(root + "/images", files={"file": ("sample.png", image_bytes())}, data={"relative_path": "folder/sample.png"})
    assert response.status_code == 413, response.text
    assert client.get(root).json()["images"] == []
    assert len(list(objects.list("projects/"))) == 2
    assert_accounted(settings, db)
    exact = replace(settings, storage_limit_bytes=current_used)
    limited = TestClient(create_app(exact, database=db, objects=objects), base_url=settings.public_url)
    limited.cookies.update(client.cookies)
    limited.headers["X-CSRF-Token"] = client.headers["X-CSRF-Token"]
    assert limited.post(root + "/categories", json={"name": "too much"}).status_code == 413
    assert limited.post("/api/v1/projects", json={"name": "too much"}).status_code == 413
    assert client.get(root).json()["categories"] == []
    assert_accounted(settings, db)


@pytest.mark.parametrize("failure", ["after_reservation", "after_replace"])
def test_recovery_applies_reserved_generation_once_after_crash(hosted, monkeypatch, failure):
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    with db.session() as session:
        old_used = session.get(User, "alice").storage_bytes
    if failure == "after_reservation":
        monkeypatch.setattr(quota, "recover_project", lambda *a, **kw: (_ for _ in ()).throw(OSError("power loss")))
    else:
        original = quota._durable_file
        def failed_fsync(path):
            if "projects" in Path(path).parts:
                raise OSError("power loss after atomic replace")
            original(path)
        monkeypatch.setattr(quota, "_durable_file", failed_fsync)
    assert client.post(root + "/categories", json={"name": "Accepted generation"}).status_code == 503
    with db.session() as session:
        user = session.get(User, "alice")
        mutation = session.scalar(select(MetadataMutation))
        assert mutation is not None
        assert user.storage_bytes == old_used
        assert user.reserved_bytes == mutation.reserved_bytes > 0
        assert session.get(Project, project["id"]).pending_mutation_id == mutation.id
    monkeypatch.undo()
    # Any subsequent read settles the journal before returning a project.
    recovered = client.get(root)
    assert recovered.status_code == 200, recovered.text
    assert [c["name"] for c in recovered.json()["categories"]] == ["Accepted generation"]
    assert_accounted(settings, db)
    with db.session() as session:
        charged = session.get(User, "alice").storage_bytes
    assert client.get(root).status_code == 200
    with db.session() as session:
        assert session.get(User, "alice").storage_bytes == charged


def test_legacy_migration_counts_metadata_and_thumbnails_once(hosted):
    settings, db, objects, app, clients = hosted
    project = project_and_image(clients["alice"])
    with db.session() as session:
        record = session.get(Project, project["id"])
        image = session.scalar(select(ImageObject).where(ImageObject.project_id == record.id))
        session.get(User, "alice").storage_bytes = image.size_bytes
        session.get(ResourceCounter, "global").storage_bytes = image.size_bytes
        record.metadata_bytes, record.quota_version = 0, 0
        image.thumbnail_bytes = 0
    quota.recover_project(settings, db, project["id"], objects)
    quota.recover_project(settings, db, project["id"], objects)
    assert_accounted(settings, db)


def test_rejected_state_preserves_previous_revision_and_masks(hosted):
    settings, db, objects, app, clients = hosted
    client = clients["alice"]
    project = project_and_image(client)
    root = f"/api/v1/projects/{project['id']}"
    category = client.post(root + "/categories", json={"name": "Carrot"}).json()
    before = client.get(root + "/images/1/state").json()
    with db.session() as session:
        limit = session.get(User, "alice").storage_bytes
    limited = TestClient(create_app(replace(settings, storage_limit_bytes=limit), database=db, objects=objects), base_url=settings.public_url)
    limited.cookies.update(client.cookies)
    limited.headers["X-CSRF-Token"] = client.headers["X-CSRF-Token"]
    candidate = dict(before)
    candidate["annotations"] = [{"id": "new", "category_id": category["id"], "mask": {"size": [12, 16], "counts": [0, 192]}}]
    response = limited.put(root + "/images/1/state", json=candidate)
    assert response.status_code == 413, response.text
    assert client.get(root + "/images/1/state").json() == before
    assert_accounted(settings, db)


def test_mutation_directory_entries_are_durable_before_journal_commit(hosted, monkeypatch):
    from contextlib import contextmanager
    settings, db, objects, app, clients = hosted
    directory_syncs = []
    original_tree, original_session = quota._durable_directory_tree, db.session

    def tracked_tree(config, directory):
        original_tree(config, directory)
        directory_syncs.append(Path(directory))

    @contextmanager
    def verified_commit():
        with original_session() as session:
            yield session
            for mutation in session.new:
                if isinstance(mutation, MetadataMutation):
                    assert quota.mutation_directory(settings, mutation.id) in directory_syncs
            for mutation in session.deleted:
                if isinstance(mutation, MetadataMutation):
                    assert settings.data_dir / "projects" / mutation.project_id in directory_syncs

    monkeypatch.setattr(quota, "_durable_directory_tree", tracked_tree)
    monkeypatch.setattr(db, "session", verified_commit)
    response = clients["alice"].post("/api/v1/projects", json={"name": "Durable project"})
    assert response.status_code == 201, response.text
    assert_accounted(settings, db)


def test_directory_fsync_includes_all_new_parent_entries(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import os
    synced = []
    descriptors = {}
    original_open, original_fsync = os.open, os.fsync

    def tracked_open(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        descriptors[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        synced.append(descriptors[fd])
        original_fsync(fd)

    monkeypatch.setattr(quota.os, "open", tracked_open)
    monkeypatch.setattr(quota.os, "fsync", tracked_fsync)
    settings = SimpleNamespace(data_dir=tmp_path / "new-data")
    leaf = settings.data_dir / "mutations" / "generation"
    quota._durable_directory_tree(settings, leaf)
    assert synced == [leaf, leaf.parent, settings.data_dir, tmp_path]
