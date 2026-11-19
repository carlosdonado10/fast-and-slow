"""A dropped Colab session must cost minutes, not hours: runs resume from their last checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from fast_and_slow.agents.classic_rl import train_ppo
from fast_and_slow.agents.kev_rl import KevRLTrainer
from fast_and_slow.agents.policy import FakePolicy
from fast_and_slow.config import StudyConfig
from fast_and_slow.notes.bank import NoteBank


def _trainer(cfg: StudyConfig, bank: NoteBank, updates: int) -> KevRLTrainer:
    k = cfg.kev.model_copy(update={"rl_updates": updates, "checkpoint_every_updates": 1, "eval_every_updates": 2})
    return KevRLTrainer(cfg.world, bank, 2, FakePolicy(seed=5), k, seed=9)


def _at(ckpt: Path, update: int) -> Path:
    (folder,) = ckpt.glob(f"u{update:04d}-*")
    return folder


def test_interrupted_kev_rl_resumes_to_the_same_weights(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    straight = _trainer(cfg, bank, 4)
    straight.train(lambda p: 0.0)

    ckpt = tmp_path / "ckpt"
    first = _trainer(cfg, bank, 2)  # the session dies after update 2
    first.train(lambda p: 0.0, checkpoint=ckpt)
    assert json.loads((ckpt / "latest.json").read_text())["folder"] == _at(ckpt, 2).name
    # the previous checkpoint stays (in case the newest never fully reaches Drive); older ones are removed
    assert sorted(f.name[:5] for f in ckpt.glob("u*")) == ["u0001", "u0002"]

    resumed = _trainer(cfg, bank, 4)  # a new session, same command
    log = resumed.train(lambda p: 0.0, checkpoint=ckpt)
    assert resumed.decisions == straight.decisions
    assert torch.equal(resumed.policy.state_proj, straight.policy.state_proj)  # type: ignore[attr-defined]
    assert any(u.get("event", "").startswith("resumed from update 2") for u in log.updates)  # type: ignore[attr-defined]
    assert [p["decisions"] for p in log.points] == [0, straight.decisions // 2, straight.decisions]


def test_a_half_written_checkpoint_falls_back_to_the_last_complete_one(
    tmp_path: Path, cfg: StudyConfig, bank: NoteBank
) -> None:
    ckpt = tmp_path / "ckpt"
    _trainer(cfg, bank, 2).train(lambda p: 0.0, checkpoint=ckpt)
    # the next save got as far as its folder before the session died: no trainer.json, latest.json not yet moved
    (ckpt / "u0003" / "policy").mkdir(parents=True)
    t = _trainer(cfg, bank, 4)
    resumed = t.load_checkpoint(ckpt)
    assert resumed is not None and resumed[0] == 2
    # a lost or garbled pointer: the highest complete folder still wins
    (ckpt / "latest.json").write_text("{not json")
    resumed = _trainer(cfg, bank, 4).load_checkpoint(ckpt)
    assert resumed is not None and resumed[0] == 2
    (ckpt / "latest.json").unlink()
    resumed = _trainer(cfg, bank, 4).load_checkpoint(ckpt)
    assert resumed is not None and resumed[0] == 2


def test_a_checkpoint_with_a_file_missing_on_drive_falls_back_to_the_previous_one(
    tmp_path: Path, cfg: StudyConfig, bank: NoteBank, caplog: pytest.LogCaptureFixture
) -> None:
    # What Colab did: trainer.json and latest.json reached Drive, the weights file never did.
    ckpt = tmp_path / "ckpt"
    _trainer(cfg, bank, 3).train(lambda p: 0.0, checkpoint=ckpt)
    (_at(ckpt, 3) / "policy" / "fake_policy.pt").unlink()
    t = _trainer(cfg, bank, 4)
    resumed = t.load_checkpoint(ckpt)
    assert resumed is not None and resumed[0] == 2
    assert f"{_at(ckpt, 3).name} is incomplete, skipping it: policy/fake_policy.pt is missing" in caplog.text
    assert KevRLTrainer._complete_checkpoint(ckpt) == _at(ckpt, 2)


def test_a_partly_uploaded_file_does_not_count(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    ckpt = tmp_path / "ckpt"
    _trainer(cfg, bank, 3).train(lambda p: 0.0, checkpoint=ckpt)
    opt = _at(ckpt, 3) / "optimizer.pt"
    opt.write_bytes(opt.read_bytes()[:10])  # shorter than trainer.json says
    resumed = _trainer(cfg, bank, 4).load_checkpoint(ckpt)
    assert resumed is not None and resumed[0] == 2


def test_an_old_format_folder_without_its_weights_is_skipped(
    tmp_path: Path, cfg: StudyConfig, bank: NoteBank, caplog: pytest.LogCaptureFixture
) -> None:
    # The checkpoint that broke the H12 run: written before trainer.json listed its files, weights never uploaded.
    ckpt = tmp_path / "ckpt"
    _trainer(cfg, bank, 1).train(lambda p: 0.0, checkpoint=ckpt)
    shutil.rmtree(_at(ckpt, 1))
    folder = ckpt / "u0003"
    (folder / "policy").mkdir(parents=True)
    (folder / "trainer.json").write_text(json.dumps({"update": 3}))
    (folder / "optimizer.pt").write_bytes(b"x")
    (ckpt / "latest.json").write_text(json.dumps({"folder": "u0003", "update": 3}))
    t = _trainer(cfg, bank, 2)
    assert t.load_checkpoint(ckpt) is None
    assert "starting this run from update 0" in caplog.text
    log = t.train(lambda p: 0.0, checkpoint=ckpt)  # trains from scratch instead of crashing
    assert len([u for u in log.updates if "update" in u and "event" not in u]) == 2
    assert "policy/ has no weights file" in caplog.text
    # the stale folder is gone once two good ones exist, and no new save reused its name
    assert not folder.exists() and sorted(f.name[:5] for f in ckpt.glob("u*")) == ["u0001", "u0002"]


def test_an_unreadable_checkpoint_tries_the_one_before(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    ckpt = tmp_path / "ckpt"
    _trainer(cfg, bank, 3).train(lambda p: 0.0, checkpoint=ckpt)
    w = _at(ckpt, 3) / "policy" / "fake_policy.pt"  # right size, garbage inside
    w.write_bytes(b"\0" * w.stat().st_size)
    resumed = _trainer(cfg, bank, 4).load_checkpoint(ckpt)
    assert resumed is not None and resumed[0] == 2
    w = _at(ckpt, 2) / "policy" / "fake_policy.pt"
    w.write_bytes(b"\0" * w.stat().st_size)
    with pytest.raises(RuntimeError, match="could be read"):
        _trainer(cfg, bank, 4).load_checkpoint(ckpt)


def test_checkpoints_in_the_old_layout_still_resume(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    ckpt = tmp_path / "ckpt"
    _trainer(cfg, bank, 2).train(lambda p: 0.0, checkpoint=ckpt)
    folder = ckpt / json.loads((ckpt / "latest.json").read_text())["folder"]
    for item in folder.iterdir():  # rebuild the pre-folder layout: files directly in the checkpoint directory
        shutil.move(str(item), str(ckpt / item.name))
    for f in ckpt.glob("u*"):
        shutil.rmtree(f)
    (ckpt / "latest.json").unlink()
    t = _trainer(cfg, bank, 4)
    log = t.train(lambda p: 0.0, checkpoint=ckpt)
    assert any(u.get("event", "").startswith("resumed from update 2") for u in log.updates)  # type: ignore[attr-defined]
    assert (ckpt / "latest.json").exists() and not (ckpt / "trainer.json").exists()  # migrated on the next save


def test_a_missing_checkpoint_starts_fresh(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    t = _trainer(cfg, bank, 1)
    assert t.load_checkpoint(tmp_path / "nothing") is None


def test_interrupted_ppo_continues_its_curve(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    ckpt = tmp_path / "ppo"
    w = cfg.world
    _, first = train_ppo(w, bank, 2, "num", None, 600, 2, 3, lambda a: 1.0, checkpoint=ckpt, save_every=200)
    assert (ckpt / "model.zip").exists() and (ckpt / "curve.json").exists()
    model, curve = train_ppo(w, bank, 2, "num", None, 1_200, 2, 3, lambda a: 2.0, checkpoint=ckpt, save_every=200)
    assert curve[: len(first)] == first  # history kept, not re-evaluated
    assert curve[-1]["score"] == 2.0 and model.num_timesteps >= 1_200
    assert [p["decisions"] for p in curve] == sorted(p["decisions"] for p in curve)
    assert curve[-1]["seconds"] >= first[-1]["seconds"]


def test_micro_batches_give_the_same_update_as_one_pass(cfg: StudyConfig, bank: NoteBank) -> None:
    """Gradient accumulation must not change the maths, only the memory."""
    after = []
    for micro in (cfg.kev.rl_minibatch, 1):
        k = cfg.kev.model_copy(update={"rl_micro_batch": micro})
        tr = KevRLTrainer(cfg.world, bank, 4, FakePolicy(seed=0), k, seed=0, train_seed_range=cfg.eval.eval_seed)
        tr.update(tr.rollout())
        after.append([p.detach().clone() for g in tr.policy.param_groups() for p in g["params"]])
    for a, b in zip(*after, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
