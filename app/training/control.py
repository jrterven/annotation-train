"""Mode adapters for revocable outbound agents. Tokens never enter task payloads."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import secrets
import time

from fastapi import HTTPException
from .catalog import catalog
from .coordinator import device_rows
from .service import identity
from .store import LocalRepository


def token_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def validate_enrollment(body):
    if set(body.models) - {m["id"] for m in catalog()}:
        raise ValueError("Agent advertised an unsupported model.")
    if len({d.slot for d in body.devices}) != len(body.devices):
        raise ValueError("GPU slots must be unique.")


class LocalControl:
    def __init__(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.repo = LocalRepository(root)

    def pair(self, backend):
        code = secrets.token_urlsafe(24)
        project_id = token_hash(str(backend.project.directory.resolve()))
        with self.repo.transaction() as records:
            records.put(
                "project",
                project_id,
                {"id": project_id, "directory": str(backend.project.directory.resolve())},
            )
            records.put(
                "pairing",
                token_hash(code),
                {"project_id": project_id, "expires": time.time() + 600},
            )
        return {"code": code}

    def enroll(self, body):
        validate_enrollment(body)
        token = secrets.token_urlsafe(48)
        with self.repo.transaction() as records:
            key = token_hash(body.code)
            pairing = records.get("pairing", key)
            if not pairing or pairing["expires"] < time.time():
                raise HTTPException(401, "Pairing code expired or already used")
            records.delete("pairing", key)
            previous = next(
                (
                    n
                    for n in records.list("node")
                    if body.previous_token
                    and n["token_hash"] == token_hash(body.previous_token)
                    and n["project_id"] == pairing["project_id"]
                ),
                None,
            )
            if previous and any(
                lease["until"] > time.time() for lease in previous.get("leases", {}).values()
            ):
                raise ValueError(
                    "Stop the agent and wait for its leases to expire before reconnecting"
                )
            node = {
                **body.model_dump(exclude={"code", "previous_token"}),
                "id": previous["id"] if previous else identity(),
                "project_id": pairing["project_id"],
                "token_hash": token_hash(token),
                "seen_at": time.time(),
                "leases": {},
            }
            records.put("node", node["id"], node)
        return {"token": token, "node_id": node["id"], "protocol": 1}

    def node(self, token):
        with self.repo.transaction() as records:
            node = next(
                (
                    n
                    for n in records.list("node")
                    if n["token_hash"] == token_hash(token) and not n.get("revoked")
                ),
                None,
            )
        if not node:
            raise HTTPException(401, "Agent revoked or unknown")
        return node

    def projects(self, node):
        return [node["project_id"]]

    def backend(self, node, project_id):
        if project_id != node["project_id"]:
            raise HTTPException(404, "Project not found")
        from app.storage import ProjectStore
        from .local import LocalBackend

        with self.repo.transaction() as records:
            project = records.get("project", project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        return LocalBackend(
            ProjectStore(project["directory"]), lambda: self.devices(project["directory"])
        )

    @contextmanager
    def guard(self, project_records, token):
        with self.repo.transaction() as records:
            node = next(
                (
                    n
                    for n in records.list("node")
                    if n["token_hash"] == token_hash(token) and not n.get("revoked")
                ),
                None,
            )
            if not node:
                raise HTTPException(401, "Agent revoked or unknown")
            yield node
            records.put("node", node["id"], node)

    def devices(self, directory):
        project_id = token_hash(str(Path(directory).resolve()))
        with self.repo.transaction() as records:
            return device_rows(
                [
                    n
                    for n in records.list("node")
                    if n["project_id"] == project_id and not n.get("revoked")
                ]
            )

    def touch(self, token):
        with self.guard(None, token) as node:
            node["seen_at"] = time.time()
        return {"ok": True}

    def revoke(self, backend, node_id):
        with self.repo.transaction() as records:
            node = records.get("node", node_id)
            if not node or node["project_id"] != token_hash(
                str(backend.project.directory.resolve())
            ):
                raise HTTPException(404, "Agent not found")
            node["revoked"] = True
            records.put("node", node_id, node)
        return {"ok": True}

    def toggle(self, backend, node_id, body):
        with self.repo.transaction() as records:
            node = records.get("node", node_id)
            if (
                not node
                or node["project_id"] != token_hash(str(backend.project.directory.resolve()))
                or node.get("revoked")
            ):
                raise HTTPException(404, "Agent not found")
            device = next((d for d in node["devices"] if d["slot"] == body.slot), None)
            if not device:
                raise HTTPException(404, "Device not found")
            device["enabled"] = body.enabled
            records.put("node", node_id, node)
        return {"ok": True}
