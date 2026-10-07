"""Authenticate and reserve storage before Starlette parses any upload body."""
import re
import secrets
from urllib.parse import urlsplit
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from .auth import COOKIE, digest
from .models import AuthSession, utcnow
from .storage import reserve_upload, release_reservation


class BodyTooLarge(Exception):
    pass


class HostedMiddleware:
    def __init__(self, app, db, settings):
        self.app, self.db, self.settings = app, db, settings

    def session_identity(self, raw_session):
        with self.db.session() as session:
            auth = session.get(AuthSession, digest(raw_session))
            if auth and auth.expires_at > utcnow():
                return {"user_id": auth.user_id, "csrf_token": auth.csrf_token}
        return {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope, receive)
        state = scope.setdefault("state", {})
        state.update(user_id=None, csrf_token=None, reservation_id=None)
        path = scope["path"]
        api = path.startswith("/api/")
        # Public legal pages and immutable frontend assets remain available
        # even while the account database is restarting.
        raw_session = request.cookies.get(COOKIE)
        if api and raw_session:
            state.update(await run_in_threadpool(self.session_identity, raw_session))
        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode()
        if api and origin and origin.rstrip("/") != self.settings.public_url.rstrip("/"):
            return await JSONResponse({"detail": "Origin not allowed"}, status_code=403)(scope, receive, send)
        if api and scope["method"] in {"POST", "PUT", "PATCH", "DELETE"}:
            if not state["user_id"]:
                return await JSONResponse({"detail": "Sign in to continue"}, status_code=401)(scope, receive, send)
            csrf = headers.get(b"x-csrf-token", b"").decode()
            if not secrets.compare_digest(csrf, state["csrf_token"]):
                return await JSONResponse({"detail": "Invalid CSRF token"}, status_code=403)(scope, receive, send)
        maximum = self.settings.max_upload_bytes + 64 * 1024
        length_header = headers.get(b"content-length")
        try:
            declared = int(length_header) if length_header else maximum
            if declared < 0 or declared > maximum:
                raise BodyTooLarge()
            upload = re.fullmatch(r"/api/v1/projects/([a-f0-9-]{36})/images/?", path)
            if upload and scope["method"] == "POST":
                state["reservation_id"] = await run_in_threadpool(reserve_upload, self.db, self.settings,
                    state["user_id"], upload[1], min(declared, self.settings.max_upload_bytes))
        except BodyTooLarge:
            return await JSONResponse({"detail": "Request exceeds upload limit"}, status_code=413)(scope, receive, send)
        except HTTPException as exc:
            return await JSONResponse({"detail": exc.detail}, status_code=exc.status_code)(scope, receive, send)
        except ValueError:
            return await JSONResponse({"detail": "Invalid content length"}, status_code=400)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > maximum or (length_header and received > declared):
                    raise BodyTooLarge()
            return message

        async def safe_send(message):
            if message["type"] == "http.response.start":
                additions = [(b"x-content-type-options", b"nosniff"),
                             (b"referrer-policy", b"same-origin"), (b"x-frame-options", b"DENY")]
                if api:
                    additions.append((b"cache-control", b"private, no-store"))
                message["headers"] = [(k, v) for k, v in message.get("headers", [])
                                      if k.lower() not in {item[0] for item in additions}] + additions
            await send(message)
        try:
            await self.app(scope, limited_receive, safe_send)
        except BodyTooLarge:
            await JSONResponse({"detail": "Request exceeds upload limit"}, status_code=413)(scope, receive, safe_send)
        finally:
            if state["reservation_id"]:
                await run_in_threadpool(release_reservation, self.db, state["reservation_id"])
