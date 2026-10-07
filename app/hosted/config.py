"""Fail-closed server configuration. Never serialize this object into an API response."""
from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    public_url: str
    google_client_id: str = field(repr=False)
    google_client_secret: str = field(repr=False)
    r2_endpoint_url: str = field(repr=False)
    r2_access_key_id: str = field(repr=False)
    r2_secret_access_key: str = field(repr=False)
    environment: str = "dev"
    r2_bucket: str = "annotation-dev"
    data_dir: Path = Path(".hosted-data")
    cache_dir: Path = Path(".hosted-cache")
    primary_worker_url: str = ""
    primary_worker_token: str = field(default="", repr=False)
    fallback_worker_url: str = ""
    fallback_worker_token: str = field(default="", repr=False)
    storage_limit_bytes: int = 1_000_000_000
    global_storage_limit_bytes: int = 50_000_000_000
    max_request_bytes: int = 64 * 1024 * 1024
    max_mask_pixels_per_request: int = 1_024_000_000
    cache_limit_bytes: int = 5_000_000_000
    inference_limit: int = 300
    max_pending_user: int = 2
    max_pending_global: int = 64
    max_runtime_seconds: int = 120
    worker_grace_seconds: int = 5
    worker_poll_seconds: int = 5
    dispatcher_poll_milliseconds: int = 200
    session_seconds: int = 7 * 86400
    min_free_disk_bytes: int = 100_000_000_000

    @property
    def secure_cookies(self):
        return urlsplit(self.public_url).scheme == "https"

    @classmethod
    def from_env(cls):
        required = ["DATABASE_URL", "PUBLIC_URL", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
                    "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
        values = {}
        for key in required:
            value = os.environ.get(f"ANNOTATION_{key}", "").strip()
            if not value:
                raise ValueError(f"Missing ANNOTATION_{key}")
            values[key.lower()] = value
        env = os.environ.get("ANNOTATION_ENVIRONMENT", "dev")
        values.update(environment=env, r2_bucket=os.environ.get("ANNOTATION_R2_BUCKET", f"annotation-{env}"))
        for name in ("data_dir", "cache_dir"):
            if value := os.environ.get("ANNOTATION_" + name.upper()):
                values[name] = Path(value)
        for name in ("primary_worker_url", "primary_worker_token", "fallback_worker_url", "fallback_worker_token"):
            values[name] = os.environ.get("ANNOTATION_" + name.upper(), "")
        for item in fields(cls):
            if item.type is int and (value := os.environ.get("ANNOTATION_" + item.name.upper())):
                values[item.name] = int(value)
        return cls(**values)

    def validate(self):
        for item in fields(self):
            if item.type is int and getattr(self, item.name) < (0 if item.name == "min_free_disk_bytes" else 1):
                raise ValueError(f"Invalid {item.name}")
        if self.environment not in {"dev", "prod"}:
            raise ValueError("Environment must be dev or prod")
        if self.r2_bucket != f"annotation-{self.environment}":
            raise ValueError("The R2 bucket must match the configured environment")
        parsed = urlsplit(self.public_url)
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("PUBLIC_URL must be an origin")
        if parsed.scheme != "https" and not (self.environment == "dev" and parsed.scheme == "http" and parsed.hostname == "localhost"):
            raise ValueError("HTTPS is required except for explicit localhost development")
        if not self.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("Hosted mode requires PostgreSQL")
        endpoint = urlsplit(self.r2_endpoint_url)
        if endpoint.scheme != "https" or not endpoint.hostname or not endpoint.hostname.endswith(".r2.cloudflarestorage.com"):
            raise ValueError("A Cloudflare R2 HTTPS endpoint is required")
        if not all([self.google_client_id, self.google_client_secret, self.r2_access_key_id, self.r2_secret_access_key]):
            raise ValueError("OAuth and R2 credentials are required")
        for prefix in ("primary", "fallback"):
            url, token = getattr(self, f"{prefix}_worker_url"), getattr(self, f"{prefix}_worker_token")
            if bool(url) != bool(token):
                raise ValueError("Each configured worker needs a URL and credential")
            if token and len(token) < 32:
                raise ValueError("Worker credentials must contain at least 32 characters")
