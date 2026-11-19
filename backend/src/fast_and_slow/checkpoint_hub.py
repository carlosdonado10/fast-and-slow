"""A copy of every Kev RL checkpoint in a private Hugging Face repo, confirmed before training continues.

Google Drive's Colab mount uploads in the background and reports nothing: four sessions in a row lost checkpoints that
had been written (files missing, half-deleted folders) because the runtime ended before Drive caught up. A Hub upload
is a single commit that either lands whole or fails loudly, and `push` returns only after it landed. So the Drive copy
stays (it is what `fns status` reads), but the Hub copy is the one a new session can rely on.

One commit per save: add the new folder, update `latest.json`, delete all but the previous folder. A session that
dies mid-upload leaves the repo exactly as it was after the last save.

Enabled by FNS_CHECKPOINT_REPO (e.g. `you/fast-and-slow-checkpoints`) and an HF_TOKEN with write access.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .progress import log

ENV = "FNS_CHECKPOINT_REPO"


class HubMirror:
    def __init__(self, repo_id: str, prefix: str, api: Any = None, download: Callable[..., Any] | None = None):
        """api: an HfApi (tests pass a stand-in); download: huggingface_hub.snapshot_download."""
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        if download is None:
            from huggingface_hub import snapshot_download

            download = snapshot_download
        self.download: Callable[..., Any] = download
        self.repo_id, self.prefix, self.api = repo_id, prefix.strip("/"), api

    @classmethod
    def from_env(cls, prefix: str) -> HubMirror | None:
        repo = os.environ.get(ENV, "").strip()
        return cls(repo, prefix) if repo else None

    def check(self) -> None:
        """Fail before any GPU time is spent if the repo can't be written (no token, or a read-only one)."""
        try:
            self.api.create_repo(self.repo_id, private=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"can't use {self.repo_id} for checkpoints ({type(e).__name__}: {e}). The HF_TOKEN secret needs "
                "write access: huggingface.co/settings/tokens > Create new token > Write."
            ) from e

    def _remote_folders(self) -> list[str]:
        try:
            entries = self.api.list_repo_tree(self.repo_id, path_in_repo=self.prefix)
            names = [str(e.path).rsplit("/", 1)[-1] for e in entries]
        except Exception as e:  # nothing pushed yet for this run: the path doesn't exist
            if "404" in str(e) or "not found" in str(e).lower() or type(e).__name__.endswith("NotFoundError"):
                return []
            raise
        return sorted(n for n in names if n.startswith("u"))

    def push(self, folder: Path, keep: str | None) -> None:
        """Upload `folder` and point latest.json at it; remove remote folders other than it and `keep`."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        t0 = time.monotonic()
        ops: list[Any] = [
            CommitOperationAdd(
                path_in_repo=f"{self.prefix}/{folder.name}/{f.relative_to(folder).as_posix()}", path_or_fileobj=str(f)
            )
            for f in sorted(folder.rglob("*"))
            if f.is_file()
        ]
        pointer = json.dumps({"folder": folder.name}).encode()
        ops.append(CommitOperationAdd(path_in_repo=f"{self.prefix}/latest.json", path_or_fileobj=pointer))
        ops += [
            CommitOperationDelete(path_in_repo=f"{self.prefix}/{old}/", is_folder=True)
            for old in self._remote_folders()
            if old not in (folder.name, keep)
        ]
        self.api.create_commit(self.repo_id, operations=ops, commit_message=f"{self.prefix}: {folder.name}")
        log.info("checkpoint %s confirmed on the Hub (%s) in %.0fs", folder.name, self.repo_id, time.monotonic() - t0)

    def pull(self, dest: Path, is_complete: Callable[[Path], bool]) -> None:
        """Copy the Hub's checkpoint folders into `dest` where dest's copy is missing or incomplete."""
        missing = [n for n in self._remote_folders() if not is_complete(dest / n)]
        if not missing:
            return
        with tempfile.TemporaryDirectory() as tmp:
            self.download(self.repo_id, allow_patterns=[f"{self.prefix}/{n}/**" for n in missing], local_dir=tmp)
            for n in missing:
                src = Path(tmp) / self.prefix / n
                if src.is_dir():
                    shutil.rmtree(dest / n, ignore_errors=True)
                    shutil.copytree(src, dest / n)
                    log.info("checkpoint %s fetched from the Hub", n)
