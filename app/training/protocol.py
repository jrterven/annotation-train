"""Version-one training agent protocol. All payloads are bounded and declarative."""

from typing import Literal
from pydantic import Field
from .data import Strict

PROTOCOL_VERSION = 1
LEASE_SECONDS = 90
HEARTBEAT_SECONDS = 10


class Device(Strict):
    slot: str = Field(pattern=r"^[0-9]{1,3}$")
    name: str = Field(min_length=1, max_length=120)
    enabled: bool = True
    identity: str = Field(default="", max_length=64)


class Enrollment(Strict):
    code: str = Field(min_length=20, max_length=100)
    name: str = Field(min_length=1, max_length=80)
    devices: list[Device] = Field(min_length=1, max_length=32)
    models: list[str] = Field(max_length=12)
    runtime: dict[str, str] = Field(max_length=12)
    protocol: Literal[1] = 1
    previous_token: str | None = Field(default=None, max_length=100)


class Claim(Strict):
    slot: str = Field(pattern=r"^[0-9]{1,3}$")


class ClassMetric(Strict):
    category_id: int
    map: float | None = Field(default=None, ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)


class Metric(Strict):
    epoch: int = Field(ge=1, le=10000)
    train_loss: float | None = None
    validation_loss: float | None = None
    map: float | None = Field(default=None, ge=0, le=1)
    box_map: float | None = Field(default=None, ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    elapsed: float | None = Field(default=None, ge=0)
    per_class: list[ClassMetric] = Field(default_factory=list, max_length=10000)


class Update(Strict):
    project_id: str = Field(max_length=160)
    kind: Literal["run", "operation"]
    work_id: str = Field(max_length=80)
    attempt_id: str = Field(max_length=80)
    sequence: int = Field(ge=0)
    metric: Metric | None = None
    progress: float | None = Field(default=None, ge=0, le=1)
    status: Literal["running", "paused", "completed", "failed", "cancelled", "syncing"] = "running"
    checkpoint: bool = False
    error: str | None = Field(default=None, max_length=500)
    result: dict | None = None


class ArtifactInput(Strict):
    project_id: str = Field(max_length=160)
    work_id: str = Field(max_length=80)
    attempt_id: str = Field(max_length=80)
    size_bytes: int = Field(gt=0, le=8 * 1024**3)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    name: Literal["model.pt", "model.pth", "manifest.json"]


class PredictionInput(Strict):
    purpose: Literal["preview", "preannotate"] = "preannotate"
    image_ids: list[int] = Field(min_length=1, max_length=1000)
    confidence: float = Field(default=0.5, ge=0, le=1)


class DeviceToggle(Strict):
    slot: str = Field(pattern=r"^[0-9]{1,3}$")
    enabled: bool
