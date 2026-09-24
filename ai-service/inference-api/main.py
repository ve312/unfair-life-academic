"""Entry point for the Unfair Life inference API.

Two equivalent ways to start it, from the inference-api folder:

    python main.py
    uvicorn main:app --host 0.0.0.0 --port $PORT      (Render start command)

Both bind to 0.0.0.0 and listen on PORT (Render injects it; 8000 locally).
Interactive docs: http://localhost:8000/docs

Both also read an optional inference-api/.env (see load_env_file), like the
BFF and the CMS do locally, so BFF_SECRET_TOKEN can be shared by all three.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Loads KEY=VALUE lines from a local .env into os.environ.

    Variables already set in the process environment win, so on Render (which
    has no .env) the dashboard values are used unchanged. Blank lines and
    '#' comments are skipped, and surrounding quotes are stripped.

    Args:
        path: The .env file; nothing happens when it does not exist.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


load_env_file()

import uvicorn  # noqa: E402

from src.infrastructure.web.fast_api_app import app  # noqa: E402

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
