#!/usr/bin/env python3
"""Launch the CPU-only hosted application using private environment settings."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Hosted annotation server (Google OAuth + private R2)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    from app.hosted.config import Settings
    from app.hosted.main import create_app
    import uvicorn
    settings = Settings.from_env()
    settings.validate()
    if not (Path(__file__).resolve().parent / "frontend/dist/index.html").is_file():
        raise SystemExit("Build the frontend first: cd frontend && npm ci && npm run build")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
