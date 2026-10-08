"""Bounded, short-lived worker memory; never persisted with the attempt ledger."""
from collections import OrderedDict
import base64
import binascii
import hashlib
import time


class WorkerImageCache:
    def __init__(self, max_bytes=512 * 1024 * 1024, ttl=1800, clock=time.monotonic):
        self.max_bytes, self.ttl, self.clock = max_bytes, ttl, clock
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def discard(self, key):
        encoded, _ = self.entries.pop(key)
        self.bytes -= len(encoded)

    def prune(self):
        now = self.clock()
        for key, (_, used) in list(self.entries.items()):
            if now - used >= self.ttl:
                self.discard(key)

    def get(self, key):
        self.prune()
        if key not in self.entries:
            self.misses += 1
            return None
        encoded, _ = self.entries[key]
        self.entries[key] = (encoded, self.clock())
        self.entries.move_to_end(key)
        self.hits += 1
        return encoded

    def put(self, key, sha256, encoded):
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid image encoding") from exc
        if not raw or hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError("Image checksum mismatch")
        self.prune()
        if key in self.entries:
            self.discard(key)
        # Oversized originals still run; they simply do not occupy the cache.
        if len(encoded) > self.max_bytes:
            return
        while self.entries and (self.bytes + len(encoded) > self.max_bytes or len(self.entries) >= 32):
            self.discard(next(iter(self.entries)))
        self.entries[key] = (encoded, self.clock())
        self.bytes += len(encoded)

    def stats(self):
        return {"entries": len(self.entries), "bytes": self.bytes, "hits": self.hits, "misses": self.misses}


class DecodedImageCache:
    """Model-process LRU: avoid decoding the same immutable original per click."""
    def __init__(self, max_bytes=256 * 1024 * 1024, ttl=1800, clock=time.monotonic):
        self.max_bytes, self.ttl, self.clock = max_bytes, ttl, clock
        self.entries = OrderedDict()
        self.bytes = 0

    def discard(self, key):
        _, _, size, _ = self.entries.pop(key)
        self.bytes -= size

    def prune(self):
        for key, (_, _, _, used) in list(self.entries.items()):
            if self.clock() - used >= self.ttl:
                self.discard(key)

    def resolve(self, request):
        from .job_protocol import decode_image
        self.prune()
        key, encoded = request.image_key, request.image_base64
        cached = self.entries.get(key)
        # Check the actual encoded content too; this boundary must never trust
        # a caller-supplied digest for different bytes, even on a cache hit.
        if cached is not None and cached[0] == encoded:
            self.entries[key] = (*cached[:3], self.clock())
            self.entries.move_to_end(key)
            return cached[1]
        image, digest = decode_image(encoded)
        if digest != request.sha256:
            raise ValueError("Image checksum does not match its immutable identity.")
        size = image.width * image.height * 4 + len(encoded)
        if key in self.entries:
            self.discard(key)
        if size <= self.max_bytes:
            while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= 4):
                self.discard(next(iter(self.entries)))
            self.entries[key] = (encoded, image, size, self.clock())
            self.bytes += size
        return image
