#!/usr/bin/env python3
"""One explicit setup command for Python dependencies and the web client."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Run this installer with Python 3.12: python3.12 scripts/setup.py")
    npm = shutil.which("npm")
    if not npm:
        raise SystemExit("Install Node.js 22.x (22.13 or newer), 24.x or 26+, and npm before continuing.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.create(env, with_pip=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True, cwd=ROOT)
    subprocess.run([npm, "ci"], check=True, cwd=ROOT / "frontend")
    subprocess.run([npm, "run", "build"], check=True, cwd=ROOT / "frontend")
    print("\nInstallation complete. Start with: python3 run.py")


if __name__ == "__main__":
    main()
