"""OS-level reservations shared by local SAM and the optional training supervisor."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess


@contextmanager
def reserve_gpu(device):
    key = hashlib.sha256(device["uuid"].encode()).hexdigest()
    path = (
        Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / f"annotation-gpu-{os.getuid()}-{key}.lock"
    )
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "GPU is reserved. Release SAM or wait for its training job to finish."
            ) from error
        yield


def reserve_sam_gpu():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    return reserve_gpu({"uuid": uuid})
