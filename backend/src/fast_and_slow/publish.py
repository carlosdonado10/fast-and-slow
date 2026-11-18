"""Send a finished report to the runs API, as a new run. Standard library only, so it works from Colab as is."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def git_commit(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def gpu_name(root: Path) -> str:
    speed = root / "speed.json"
    return str(json.loads(speed.read_text()).get("gpu", "")) if speed.exists() else ""


def post_run(api: str, token: str, body: dict[str, Any], timeout: float = 60.0) -> str:
    """POST /api/runs; returns the new run's id. Raises RuntimeError with the server's reason on failure."""
    url = api.rstrip("/") + "/runs"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return str(json.loads(res.read())["id"])
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except (ValueError, AttributeError):
            detail = ""
        raise RuntimeError(f"{url} refused the run (HTTP {e.code}): {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"couldn't reach {url}: {e.reason}. Is the stack running and the URL right?") from e
