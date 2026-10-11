"""Local project adapter: independent metadata and content-addressed originals."""

from pathlib import Path
import hashlib
import os
import shutil
import tempfile

from .data import capture
from .store import LocalRepository


class LocalBackend:
    hosted = False

    def __init__(self, project, devices=None):
        self.project = project
        self.repo = LocalRepository(project.directory)
        self._devices = devices or (lambda: [])

    def capture(self):
        return capture(self.project)

    def devices(self):
        return self._devices()

    def retain(self, dataset):
        directory = self.repo.root / "images"
        directory.mkdir(exist_ok=True, mode=0o700)
        for image in dataset["images"]:
            destination = directory / image["sha256"]
            if destination.is_file():
                with destination.open("rb") as existing:
                    if hashlib.file_digest(existing, "sha256").hexdigest() == image["sha256"]:
                        continue
            descriptor, name = tempfile.mkstemp(dir=directory)
            try:
                with os.fdopen(descriptor, "wb") as target, self.project.image_path(
                    image["id"]
                ).open("rb") as source:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                with open(name, "rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != image["sha256"]:
                        raise ValueError("An original image changed while preparing the dataset.")
                os.replace(name, destination)
            finally:
                Path(name).unlink(missing_ok=True)

    def image(self, image):
        path = self.repo.root / "images" / image["sha256"]
        if not path.is_file():
            raise FileNotFoundError("Snapshot original is unavailable")
        return path
