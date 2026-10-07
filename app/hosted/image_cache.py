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
