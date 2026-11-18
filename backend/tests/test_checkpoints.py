"""A dropped Colab session must cost minutes, not hours: runs resume from their last checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from fast_and_slow.agents.classic_rl import train_ppo
from fast_and_slow.agents.kev_rl import KevRLTrainer
from fast_and_slow.agents.policy import FakePolicy
from fast_and_slow.config import StudyConfig
from fast_and_slow.notes.bank import NoteBank


def _trainer(cfg: StudyConfig, bank: NoteBank, updates: int) -> KevRLTrainer:
    k = cfg.kev.model_copy(update={"rl_updates": updates, "checkpoint_every_updates": 1, "eval_every_updates": 2})
    return KevRLTrainer(cfg.world, bank, 2, FakePolicy(seed=5), k, seed=9)


def test_interrupted_kev_rl_resumes_to_the_same_weights(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    straight = _trainer(cfg, bank, 4)
    straight.train(lambda p: 0.0)

    ckpt = tmp_path / "ckpt"
    first = _trainer(cfg, bank, 2)  # the session dies after update 2
    first.train(lambda p: 0.0, checkpoint=ckpt)
    assert json.loads((ckpt / "latest.json").read_text())["folder"] == "u0002"
    assert [f.name for f in ckpt.glob("u*")] == ["u0002"]  # older checkpoints removed once a newer one is complete

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


def test_checkpoints_in_the_old_layout_still_resume(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    ckpt = tmp_path / "ckpt"
    _trainer(cfg, bank, 2).train(lambda p: 0.0, checkpoint=ckpt)
    folder = ckpt / json.loads((ckpt / "latest.json").read_text())["folder"]
    for item in folder.iterdir():  # rebuild the pre-folder layout: files directly in the checkpoint directory
        shutil.move(str(item), str(ckpt / item.name))
    shutil.rmtree(folder)
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
