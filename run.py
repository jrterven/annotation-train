#!/usr/bin/env python3
"""Launch the locally installed application; never install packages implicitly."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Annotation and Training · local SAM 3 editor")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--dev", action="store_true", help="Reload the backend when its source changes")
    args = parser.parse_args()
    env_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if env_python.is_file() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        try:
            return subprocess.call([str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        except KeyboardInterrupt:
            return 130
    if not (ROOT / "frontend/dist/index.html").is_file():
        print("Build the frontend first: cd frontend && npm ci && npm run build", file=sys.stderr)
        return 1
    os.chdir(ROOT)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ["SAM3_DEVICE"] = args.device
    try:
        import uvicorn
    except ImportError:
        print("Install the dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    url = "http://127.0.0.1:8765"
    print(f"\nAnnotation and Training → {url}\nCtrl+C to stop.\n", flush=True)
    if not args.no_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8765, reload=args.dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
