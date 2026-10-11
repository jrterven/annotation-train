"""One disposable GPU subprocess. Never imported by the web application."""

import json
import os
from pathlib import Path
import sys
import signal
import threading
import time


def watch_supervisor(directory, parent):
    while True:
        try:
            supervisor = json.loads((directory / "supervisor.json").read_text())
            expired = supervisor["pid"] != parent or supervisor["deadline"] <= time.monotonic()
        except (OSError, ValueError, KeyError):
            expired = True
        if expired or os.getppid() != parent:
            # Also terminate data-loader children if the supervisor was killed.
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return
        time.sleep(2)


def main():
    directory, weights = map(Path, sys.argv[1:3])
    threading.Thread(target=watch_supervisor, args=(directory, os.getppid()), daemon=True).start()
    assignment = json.loads((directory / "assignment.json").read_text())
    work, manifest = assignment["work"], assignment["dataset"]
    last = 0.0
    with (directory / "events.jsonl").open("a", buffering=1) as events:

        def emit(event):
            nonlocal last
            now = time.monotonic()
            if set(event) == {"progress"} and now - last < 2:
                return
            last = now
            events.write(json.dumps(event, allow_nan=False) + "\n")

        from .adapters import train, predict, evaluate

        operation = work.get("operation", "train")
        if operation == "train":
            result = train(
                work, manifest, directory, weights, emit, lambda: (directory / "pause").exists()
            )
        else:
            result = {
                "status": "completed",
                "result": (evaluate if operation == "evaluate" else predict)(
                    work, manifest, directory
                ),
            }
        temporary = directory / "result.next.json"
        with temporary.open("w") as output:
            json.dump(result, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / "result.json")


if __name__ == "__main__":
    main()
