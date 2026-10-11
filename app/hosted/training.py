"""CPU-only training persistence and private dataset access."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile

from fastapi import HTTPException
from sqlalchemy import select

from app.training.data import capture, encoded
from .auth import user_for_request
from .models import Project, TrainingRecord, TrainingNode, User, ImageObject
from .quota import recover_project_locked, _snapshot
from .storage import project_store


class PostgresRecords:
    def __init__(self, session, project, counter, settings):
        self.session, self.project, self.counter, self.settings = (
            session,
            project,
            counter,
            settings,
        )

    def get(self, kind, key):
        row = self.session.get(TrainingRecord, (self.project.id, kind, key))
        return copy.deepcopy(row.value) if row else None

    def list(self, kind):
        return [
            copy.deepcopy(row.value)
            for row in self.session.scalars(
                select(TrainingRecord)
                .where(TrainingRecord.project_id == self.project.id, TrainingRecord.kind == kind)
                .order_by(TrainingRecord.id)
            )
        ]

    def put(self, kind, key, value):
        value = json.loads(encoded(value))  # no mutable ORM JSON aliasing
        row = self.session.get(TrainingRecord, (self.project.id, kind, key))
        # External artifact content is quota-accounted by its immutable record.
        size = len(encoded(value)) + (
            value.get("size_bytes", 0)
            if kind == "artifact"
            else value.get("remaining", 0) if kind == "budget" else 0
        )
        delta = size - (row.size_bytes if row else 0)
        user = self.session.get(User, self.project.user_id)
        work_id = (
            key
            if kind in {"run", "operation"}
            else value.get("work_id") if kind in {"event", "attempt"} else None
        )
        budget = (
            self.session.get(TrainingRecord, (self.project.id, "budget", work_id))
            if work_id
            else None
        )
        if budget and delta > 0:
            remaining = budget.value["remaining"]
            updated = {"remaining": max(0, remaining - delta)}
            budget_size = len(encoded(updated)) + updated["remaining"]
            released = budget.size_bytes - budget_size
            budget.value, budget.size_bytes = updated, budget_size
            user.storage_bytes -= released
            self.counter.storage_bytes -= released
        if delta > 0 and (
            user.storage_bytes + user.reserved_bytes + delta > self.settings.storage_limit_bytes
            or self.counter.storage_bytes + self.counter.reserved_bytes + delta
            > self.settings.global_storage_limit_bytes
        ):
            raise HTTPException(
                413, "Storage quota exceeded. Delete an unused training result or dataset."
            )
        user.storage_bytes += delta
        self.counter.storage_bytes += delta
        if row:
            row.value, row.size_bytes = value, size
        else:
            self.session.add(
                TrainingRecord(
                    project_id=self.project.id, kind=kind, id=key, value=value, size_bytes=size
                )
            )
        self.session.flush()

    def delete(self, kind, key):
        row = self.session.get(TrainingRecord, (self.project.id, kind, key))
        if row:
            self.session.get(User, self.project.user_id).storage_bytes -= row.size_bytes
            self.counter.storage_bytes -= row.size_bytes
            self.session.delete(row)
            self.session.flush()


class PostgresRepository:
    def __init__(self, settings, db, objects, user_id, project_id):
        self.settings, self.db, self.objects = settings, db, objects
        self.user_id, self.project_id = user_id, project_id

    @contextmanager
    def transaction(self):
        with self.db.session() as session:
            counter = self.db.global_lock(session)
            project = session.scalar(
                select(Project)
                .where(
                    Project.id == self.project_id,
                    Project.user_id == self.user_id,
                    Project.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if not project:
                raise HTTPException(404, "Project not found")
            recover_project_locked(self.settings, session, counter, project, self.objects)
            yield PostgresRecords(session, project, counter, self.settings)


class HostedBackend:
    hosted = True

    def __init__(self, settings, db, objects, user_id, project_id):
        self.settings, self.db, self.objects, self.user_id, self.project_id = (
            settings,
            db,
            objects,
            user_id,
            project_id,
        )
        self.repo = PostgresRepository(settings, db, objects, user_id, project_id)

    def capture(self):
        with tempfile.TemporaryDirectory(
            dir=self.settings.data_dir, prefix="coco-training-"
        ) as temp:
            with self.repo.transaction() as records:
                ready = {
                    im.image_id: (im.object_key, im.sha256)
                    for im in records.session.scalars(
                        select(ImageObject).where(
                            ImageObject.project_id == self.project_id, ImageObject.status == "ready"
                        )
                    )
                }
                store = project_store(self.settings, self.project_id)
                _snapshot(store.database, Path(temp) / "proyecto.sqlite3")
            from app.storage import ProjectStore

            return capture(ProjectStore(temp), ready)

    def retain(self, dataset):
        # Immutable originals are referenced, not copied. Their lifetime is the
        # project lifetime; backup manifests additionally pin these identities.
        pass

    def devices(self):
        import time

        with self.db.session() as session:
            nodes = list(
                session.scalars(
                    select(TrainingNode).where(
                        TrainingNode.user_id == self.user_id, TrainingNode.revoked.is_(False)
                    )
                )
            )
            return [
                {
                    **device,
                    "id": f"{node.id}:{device['slot']}",
                    "node_id": node.id,
                    "name": f"{node.value.get('name', 'GPU')} · {device['name']}",
                    "online": time.time() - node.value.get("seen_at", 0) < 45,
                    "models": node.value.get("models", []),
                }
                for node in nodes
                for device in node.value.get("devices", [])
            ]

    def image(self, image):
        return self.objects.open_stream(image["object_key"])


def install(app):
    from app.training.routes import project_router

    control = HostedControl(app)

    def resolve(request):
        if not app.state.settings.training_enabled:
            raise HTTPException(404, "Training is not enabled")
        user_id = user_for_request(request)
        # SSE resolves repeatedly after middleware; recheck the session itself.
        from .auth import COOKIE, digest
        from .models import AuthSession, utcnow

        with app.state.db.session() as session:
            row = session.get(AuthSession, digest(request.cookies.get(COOKIE, "")))
            if not row or row.user_id != user_id or row.expires_at <= utcnow():
                raise HTTPException(401, "Session expired")
        backend = HostedBackend(
            app.state.settings,
            app.state.db,
            app.state.objects,
            user_id,
            request.path_params["project_id"],
        )
        backend.control = control
        return backend

    app.include_router(project_router(resolve), prefix="/api/v1/projects/{project_id}/training")
    from app.training.transport import install_agent

    install_agent(
        app, control, "/api/v1/training-agent", lambda: app.state.settings.training_enabled
    )


class HostedControl:
    def __init__(self, app):
        self.app = app
        self.db, self.settings, self.objects = app.state.db, app.state.settings, app.state.objects

    def pair(self, backend):
        import secrets
        from datetime import timedelta
        from .models import TrainingPairing, utcnow
        from app.training.control import token_hash

        code = secrets.token_urlsafe(24)
        with backend.repo.transaction() as records:
            rows = list(
                records.session.scalars(
                    select(TrainingPairing).where(TrainingPairing.user_id == backend.user_id)
                )
            )
            for row in rows:
                records.session.delete(row)
            records.session.add(
                TrainingPairing(
                    code_hash=token_hash(code),
                    user_id=backend.user_id,
                    expires_at=utcnow() + timedelta(minutes=10),
                )
            )
        return {"code": code}

    def enroll(self, body):
        import secrets
        import time
        from .models import TrainingPairing, utcnow, identity
        from app.training.control import token_hash, validate_enrollment

        validate_enrollment(body)
        token = secrets.token_urlsafe(48)
        with self.db.session() as session:
            self.db.global_lock(session)
            pairing = session.get(TrainingPairing, token_hash(body.code))
            if not pairing or pairing.expires_at <= utcnow():
                raise HTTPException(401, "Pairing code expired or already used")
            if (
                len(
                    list(
                        session.scalars(
                            select(TrainingNode.id).where(
                                TrainingNode.user_id == pairing.user_id,
                                TrainingNode.revoked.is_(False),
                            )
                        )
                    )
                )
                >= 32
            ):
                raise ValueError("Disconnect an unused computer before connecting another.")
            previous = (
                session.scalar(
                    select(TrainingNode).where(
                        TrainingNode.user_id == pairing.user_id,
                        TrainingNode.token_hash == token_hash(body.previous_token),
                    )
                )
                if body.previous_token
                else None
            )
            if previous and any(
                lease["until"] > time.time() for lease in previous.value.get("leases", {}).values()
            ):
                raise ValueError(
                    "Stop the agent and wait for its leases to expire before reconnecting"
                )
            node_id = previous.id if previous else identity()
            value = {
                **body.model_dump(exclude={"code", "previous_token"}),
                "id": node_id,
                "user_id": pairing.user_id,
                "seen_at": time.time(),
                "leases": {},
            }
            if previous:
                previous.value, previous.token_hash, previous.revoked = (
                    value,
                    token_hash(token),
                    False,
                )
            else:
                session.add(
                    TrainingNode(
                        id=node_id,
                        user_id=pairing.user_id,
                        token_hash=token_hash(token),
                        value=value,
                    )
                )
            session.delete(pairing)
        return {"token": token, "node_id": node_id, "protocol": 1}

    def _row(self, session, token):
        from app.training.control import token_hash

        row = session.scalar(
            select(TrainingNode).where(
                TrainingNode.token_hash == token_hash(token), TrainingNode.revoked.is_(False)
            )
        )
        if not row:
            raise HTTPException(401, "Agent revoked or unknown")
        return row

    def node(self, token):
        with self.db.session() as session:
            return self._row(session, token).value

    def projects(self, node):
        with self.db.session() as session:
            return list(
                session.scalars(
                    select(Project.id).where(
                        Project.user_id == node["user_id"], Project.deleted_at.is_(None)
                    )
                )
            )

    def backend(self, node, project_id):
        return HostedBackend(self.settings, self.db, self.objects, node["user_id"], project_id)

    @contextmanager
    def guard(self, project_records, token):
        import copy

        row = self._row(project_records.session, token)
        value = copy.deepcopy(row.value)
        yield value
        row.value = value

    def touch(self, token):
        import copy
        import time

        with self.db.session() as session:
            self.db.global_lock(session)
            row = self._row(session, token)
            value = copy.deepcopy(row.value)
            value["seen_at"] = time.time()
            row.value = value
        return {"ok": True}

    def revoke(self, backend, node_id):
        with backend.repo.transaction() as records:
            row = records.session.get(TrainingNode, node_id)
            if not row or row.user_id != backend.user_id:
                raise HTTPException(404, "Agent not found")
            row.revoked = True
        return {"ok": True}

    def toggle(self, backend, node_id, body):
        import copy

        with backend.repo.transaction() as records:
            row = records.session.get(TrainingNode, node_id)
            if not row or row.user_id != backend.user_id or row.revoked:
                raise HTTPException(404, "Agent not found")
            value = copy.deepcopy(row.value)
            device = next((d for d in value["devices"] if d["slot"] == body.slot), None)
            if not device:
                raise HTTPException(404, "Device not found")
            device["enabled"] = body.enabled
            row.value = value
        return {"ok": True}
