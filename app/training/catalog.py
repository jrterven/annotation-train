"""Versioned model profiles; only adapters on this allowlist can execute."""

from .data import Strict
from pydantic import Field

PROFILE_VERSION = 1
FAMILIES = ("yolo26", "rfdetr")


def catalog():
    return [
        {
            "id": f"{family}-{size}-{task}",
            "family": family,
            "size": size,
            "task": task,
            "name": f"{'YOLO26' if family == 'yolo26' else 'RF-DETR'} {size.title()}",
            "profile_version": PROFILE_VERSION,
            "license": "AGPL-3.0 / Enterprise" if family == "yolo26" else "Apache-2.0",
            "license_url": (
                "https://www.ultralytics.com/license"
                if family == "yolo26"
                else "https://github.com/roboflow/rf-detr/blob/develop/LICENSE"
            ),
            "resolution": (
                640
                if family == "yolo26"
                else (
                    {"nano": 384, "small": 512, "medium": 576}
                    if task == "detection"
                    else {"nano": 312, "small": 384, "medium": 432}
                )[size]
            ),
        }
        for family in FAMILIES
        for size in ("nano", "small", "medium")
        for task in ("detection", "segmentation")
    ]


def model_profile(model_id):
    for model in catalog():
        if model["id"] == model_id:
            return model
    raise ValueError("Select a supported model from the catalog.")


class Parameters(Strict):
    epochs: int = Field(default=100, ge=1, le=10000)
    patience: int = Field(default=15, ge=1, le=10000)
    batch: int = Field(default=4, ge=1, le=512)
    resolution: int | None = Field(default=None, ge=128, le=4096)
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    augment: bool = True


class RunInput(Strict):
    dataset_id: str
    model_id: str
    device_id: str
    parameters: Parameters = Field(default_factory=Parameters)
    allow_lossy: bool = False
    request_id: str = Field(min_length=1, max_length=80)
