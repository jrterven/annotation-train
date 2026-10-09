"""Google authorization-code OIDC with server-side one-use state and sessions."""
from datetime import timedelta
import hashlib
import secrets
from urllib.parse import urlencode, urlsplit

from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.jose import JsonWebToken
from authlib.oidc.core import CodeIDToken
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
import httpx
from sqlalchemy import select

from .models import AuthFlow, AuthSession, InferenceUsage, Project, User, identity, utcnow

COOKIE = "annotation_session"
FLOW_COOKIE = "annotation_oauth"
router = APIRouter(prefix="/api/v1/auth")


def google_picture(value):
    """Keep only Google's HTTPS profile images, never arbitrary remote content."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    # Browsers treat backslashes as path separators in HTTPS URLs, unlike
    # urlsplit. Reject ambiguous input before checking the allowed hostname.
    if "\\" in value or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        return None
    try:
        url = urlsplit(value)
        if (url.scheme == "https" and url.hostname and url.hostname.endswith(".googleusercontent.com")
                and url.port in (None, 443) and not url.username and not url.password):
            return value
    except ValueError:
        pass
    return None


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def make_session(db, user_id, seconds):
    token = secrets.token_urlsafe(32)
    with db.session() as session:
        row = AuthSession(id=digest(token), user_id=user_id, csrf_token=secrets.token_urlsafe(32),
                          expires_at=utcnow() + timedelta(seconds=seconds))
        session.add(row)
    return token, row


def user_for_request(request):
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(401, "Sign in with Google to continue")
    return user_id


@router.get("/session")
def current_session(request: Request):
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        return {"user": None, "csrf_token": None, "usage": None}
    settings, db = request.app.state.settings, request.app.state.db
    with db.session() as session:
        pending = list(session.scalars(select(Project.id).where(Project.user_id == user_id,
            Project.deleted_at.is_(None), (Project.pending_mutation_id.is_not(None) | (Project.quota_version != 1)))))
    if pending:
        from .quota import recover_project
        for project_id in pending:
            recover_project(settings, db, project_id, request.app.state.objects, user_id)
    with db.session() as session:
        user = session.get(User, user_id)
        usage = session.get(InferenceUsage, (user_id, utcnow().date().isoformat()))
        return {"user": {"id": user.id, "email": user.email, "name": user.name, "picture": google_picture(user.picture)},
                "csrf_token": request.state.csrf_token,
                "usage": {"storage_bytes": user.storage_bytes,
                          "storage_limit_bytes": settings.storage_limit_bytes,
                          "inferences_used": (usage.used + usage.reserved) if usage else 0,
                          "inference_limit": settings.inference_limit}}


@router.get("/google/login")
async def google_login(request: Request):
    settings, db = request.app.state.settings, request.app.state.db
    flow_token, state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(4))
    # Explicit configured origin: never derive OAuth callbacks from untrusted Host/forwarded headers.
    callback = settings.public_url.rstrip("/") + "/api/v1/auth/google/callback"
    with db.session() as session:
        session.add(AuthFlow(id=digest(flow_token), state_hash=digest(state), nonce=nonce, verifier=verifier,
                             redirect_uri=callback, expires_at=utcnow() + timedelta(minutes=10)))
    async with AsyncOAuth2Client(settings.google_client_id, scope="openid email profile",
                                 redirect_uri=callback, code_challenge_method="S256") as client:
        url, _ = client.create_authorization_url("https://accounts.google.com/o/oauth2/v2/auth",
                                                 state=state, nonce=nonce, code_verifier=verifier)
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(FLOW_COOKIE, flow_token, max_age=600, httponly=True, secure=settings.secure_cookies,
                        samesite="lax", path="/api/v1/auth/google")
    return response


@router.get("/google/callback")
async def google_callback(request: Request):
    settings, db = request.app.state.settings, request.app.state.db
    raw_flow = request.cookies.get(FLOW_COOKIE, "")
    state = request.query_params.get("state", "")
    with db.session() as session:
        flow = session.execute(select(AuthFlow).where(AuthFlow.id == digest(raw_flow)).with_for_update()).scalar_one_or_none()
        if not flow or flow.expires_at <= utcnow() or not secrets.compare_digest(flow.state_hash, digest(state)):
            raise HTTPException(400, "Invalid or expired Google sign-in")
        # Consume before network calls, including refused consent, to prevent replay.
        session.delete(flow)
    if request.query_params.get("error"):
        response = RedirectResponse(settings.public_url.rstrip("/") + "/?error=cancelled", status_code=302)
        response.delete_cookie(FLOW_COOKIE, path="/api/v1/auth/google")
        return response
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(400, "Missing authorization code")
    try:
        async with AsyncOAuth2Client(settings.google_client_id, settings.google_client_secret,
                redirect_uri=flow.redirect_uri, token_endpoint_auth_method="client_secret_post", timeout=20) as client:
            token = await client.fetch_token("https://oauth2.googleapis.com/token", code=code,
                                            code_verifier=flow.verifier, grant_type="authorization_code")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get("https://www.googleapis.com/oauth2/v3/certs")
            response.raise_for_status()
            keys = response.json()
        claims = JsonWebToken(["RS256"]).decode(token["id_token"], keys, claims_cls=CodeIDToken,
            claims_options={"iss": {"essential": True, "values": ["https://accounts.google.com", "accounts.google.com"]},
                            "aud": {"essential": True, "value": settings.google_client_id},
                            "sub": {"essential": True}, "exp": {"essential": True}},
            claims_params={"client_id": settings.google_client_id, "nonce": flow.nonce,
                           "access_token": token.get("access_token")})
        claims.validate(leeway=30)
        if claims.get("email_verified") is not True or not claims.get("email"):
            raise ValueError("Verified email required")
    except Exception as exc:
        # Never include provider responses, codes or tokens in errors/logging.
        raise HTTPException(400, "Google sign-in could not be verified") from None
    with db.session() as session:
        db.global_lock(session)
        user = session.execute(select(User).where(User.google_sub == claims["sub"])).scalar_one_or_none()
        if not user:
            user = User(id=identity(), google_sub=claims["sub"], email=claims["email"], name=claims.get("name", claims["email"]))
            session.add(user)
        else:
            user.email, user.name = claims["email"], claims.get("name", claims["email"])
        user.picture = google_picture(claims.get("picture"))
        user_id = user.id
        old = session.get(AuthSession, digest(request.cookies.get(COOKIE, "")))
        if old:
            session.delete(old)
    raw_session, _ = make_session(db, user_id, settings.session_seconds)
    response = RedirectResponse(settings.public_url.rstrip("/") + "/", status_code=302)
    response.set_cookie(COOKIE, raw_session, max_age=settings.session_seconds, httponly=True,
                        secure=settings.secure_cookies, samesite="lax", path="/")
    response.delete_cookie(FLOW_COOKIE, path="/api/v1/auth/google")
    return response


@router.post("/logout")
def logout(request: Request):
    user_for_request(request)
    with request.app.state.db.session() as session:
        row = session.get(AuthSession, digest(request.cookies.get(COOKIE, "")))
        if row:
            session.delete(row)
    from fastapi.responses import JSONResponse
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, path="/")
    return response
