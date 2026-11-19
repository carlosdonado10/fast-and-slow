"""Kev RL checkpoints survive a lost Drive copy: each save is also committed to a Hub repo, and a resume pulls it."""

from __future__ import annotations

import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fast_and_slow.agents.kev_rl import KevRLTrainer
from fast_and_slow.agents.policy import FakePolicy
from fast_and_slow.checkpoint_hub import ENV, HubMirror
from fast_and_slow.config import StudyConfig
from fast_and_slow.notes.bank import NoteBank


@dataclass
class Add:
    path_in_repo: str
    path_or_fileobj: Any


@dataclass
class Delete:
    path_in_repo: str
    is_folder: bool = False


@pytest.fixture(autouse=True)
def fake_hub_module(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("huggingface_hub")
    mod.CommitOperationAdd = Add  # type: ignore[attr-defined]
    mod.CommitOperationDelete = Delete  # type: ignore[attr-defined]
    mod.HfApi = lambda: "api"  # type: ignore[attr-defined]
    mod.snapshot_download = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


class FakeHub:
    """A repo as a folder on disk; a commit applies all its operations or none."""

    def __init__(self, root: Path, fail_commits: int = 0, can_write: bool = True):
        self.root, self.fail_commits, self.can_write, self.commits = root, fail_commits, can_write, 0

    def create_repo(self, repo_id: str, private: bool, exist_ok: bool) -> None:
        if not self.can_write:
            raise PermissionError("403 Forbidden: token is read-only")

    def list_repo_tree(self, repo_id: str, path_in_repo: str) -> list[Any]:
        base = self.root / path_in_repo
        if not base.is_dir():
            raise FileNotFoundError(f"404 Client Error: {path_in_repo} not found")
        return [types.SimpleNamespace(path=f"{path_in_repo}/{p.name}") for p in base.iterdir()]

    def create_commit(self, repo_id: str, operations: list[Any], commit_message: str) -> None:
        if self.fail_commits:
            self.fail_commits -= 1
            raise ConnectionError("upload interrupted")
        self.commits += 1
        for op in operations:
            target = self.root / op.path_in_repo
            if isinstance(op, Delete):
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                src = op.path_or_fileobj
                target.write_bytes(src if isinstance(src, bytes) else Path(src).read_bytes())

    def download(self, repo_id: str, allow_patterns: list[str], local_dir: str) -> None:
        for pattern in allow_patterns:
            folder = pattern.removesuffix("/**")
            shutil.copytree(self.root / folder, Path(local_dir) / folder, dirs_exist_ok=True)


def _trainer(cfg: StudyConfig, bank: NoteBank, updates: int) -> KevRLTrainer:
    k = cfg.kev.model_copy(update={"rl_updates": updates, "checkpoint_every_updates": 1, "eval_every_updates": 2})
    return KevRLTrainer(cfg.world, bank, 2, FakePolicy(seed=5), k, seed=9)


def _mirror(hub: FakeHub) -> HubMirror:
    return HubMirror("me/ckpts", "full/kev_rl/H2/seed0", api=hub, download=hub.download)


def test_a_run_resumes_from_the_hub_when_drive_lost_everything(
    tmp_path: Path, cfg: StudyConfig, bank: NoteBank
) -> None:
    hub = FakeHub(tmp_path / "hub")
    straight = _trainer(cfg, bank, 4)
    straight.train(lambda p: 0.0)

    drive = tmp_path / "drive"
    _trainer(cfg, bank, 3).train(lambda p: 0.0, checkpoint=drive, mirror=_mirror(hub))
    remote = sorted(p.name[:5] for p in (hub.root / "full/kev_rl/H2/seed0").glob("u*"))
    assert remote == ["u0002", "u0003"]  # the newest and the one before; older ones deleted in the same commit

    shutil.rmtree(drive)  # what Colab's Drive did to us: nothing usable arrived
    drive.mkdir()
    (drive / "u0003-0000000000000001").mkdir()  # plus a stale, broken folder
    resumed = _trainer(cfg, bank, 4)
    log = resumed.train(lambda p: 0.0, checkpoint=drive, mirror=_mirror(hub))
    assert any(u.get("event", "").startswith("resumed from update 3") for u in log.updates)  # type: ignore[union-attr]
    assert resumed.decisions == straight.decisions


def test_a_failed_upload_keeps_training_and_the_next_save_catches_up(
    tmp_path: Path, cfg: StudyConfig, bank: NoteBank, caplog: pytest.LogCaptureFixture
) -> None:
    hub = FakeHub(tmp_path / "hub", fail_commits=1)
    _trainer(cfg, bank, 2).train(lambda p: 0.0, checkpoint=tmp_path / "drive", mirror=_mirror(hub))
    assert "is NOT on the Hub" in caplog.text and hub.commits == 1
    assert [p.name[:5] for p in (hub.root / "full/kev_rl/H2/seed0").glob("u*")] == ["u0002"]


def test_a_read_only_token_is_refused_before_any_gpu_time(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="needs write access"):
        _mirror(FakeHub(tmp_path, can_write=False)).check()
    _mirror(FakeHub(tmp_path)).check()


def test_the_mirror_is_off_unless_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert HubMirror.from_env("x") is None
    monkeypatch.setenv(ENV, "me/ckpts")
    m = HubMirror.from_env("full/kev_rl/H1/seed0")
    assert m is not None and m.repo_id == "me/ckpts" and m.prefix == "full/kev_rl/H1/seed0" and m.api == "api"


def test_listing_errors_other_than_not_found_are_raised(tmp_path: Path) -> None:
    class Down(FakeHub):
        def list_repo_tree(self, repo_id: str, path_in_repo: str) -> list[Any]:
            raise ConnectionError("hub unreachable")

    with pytest.raises(ConnectionError):
        _mirror(Down(tmp_path)).pull(tmp_path / "d", lambda p: False)
