"""The GPU bill is estimated before it's spent, and a session can stop on a time budget without losing work."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from fast_and_slow.agents.kev_rl import KevRLTrainer
from fast_and_slow.agents.policy import FakePolicy
from fast_and_slow.cli import app
from fast_and_slow.config import StudyConfig, full
from fast_and_slow.notes.bank import NoteBank
from fast_and_slow.plan import GUESS, Speed, lines, render, rl_hours, sessions_needed, updates_that_fit


def test_rl_cost_grows_with_horizon_and_updates() -> None:
    cfg, speed = full(), Speed(**GUESS)  # type: ignore[arg-type]
    d1, h1 = rl_hours(cfg, speed, 1)
    d12, h12 = rl_hours(cfg, speed, 12)
    assert d12 == 12 * d1 and h12 > h1
    assert rl_hours(cfg, speed, 12, updates=30)[1] < h12
    stages = [ln.stage for ln in lines(cfg, speed)]
    assert any(s.startswith("kev_rl H=12") for s in stages) and any(s.startswith("kev_zero") for s in stages)


def test_render_says_whether_it_fits_and_what_to_cut() -> None:
    cfg = full()
    fast = Speed(0.01, 0.02, 0.04, "H100")
    assert "Fits" in render(cfg, fast, 15, 100)
    slow = Speed(0.2, 0.3, 1.5, "T4")
    text = render(cfg, slow, 15, 100)
    fit = updates_that_fit(cfg, slow, 100 * 0.8 / 15)
    assert "Over budget" in text and f"Set kev.rl_updates to {fit}" in text and 0 < fit < cfg.kev.rl_updates
    assert "Even 1 RL update" in render(cfg, Speed(5, 5, 50, "toaster"), 15, 10)
    assert "guesses" in render(cfg, Speed(**GUESS), 15, 100)  # type: ignore[arg-type]


def test_speed_round_trips_and_defaults_to_guesses(tmp_path: Path) -> None:
    assert Speed.load(tmp_path / "none.json").gpu == GUESS["gpu"]
    Speed(0.1, 0.2, 0.3, "A100").save(tmp_path / "s.json")
    assert Speed.load(tmp_path / "s.json").gpu == "A100"


def test_deadline_pauses_at_a_checkpoint_and_resumes(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    k = cfg.kev.model_copy(update={"rl_updates": 4, "checkpoint_every_updates": 100, "eval_every_updates": 100})
    ckpt = tmp_path / "ck"
    t = KevRLTrainer(cfg.world, bank, 2, FakePolicy(seed=1), k, seed=3)
    log = t.train(lambda p: 0.0, checkpoint=ckpt, deadline=0.0)  # already out of time: stop after one update
    assert log.paused and (ckpt / "trainer.json").exists()
    t2 = KevRLTrainer(cfg.world, bank, 2, FakePolicy(seed=1), k, seed=3)
    log2 = t2.train(lambda p: 0.0, checkpoint=ckpt)
    assert not log2.paused and len([u for u in log2.updates if "loss" in u]) == 4


def test_cli_budget_commands(tmp_path: Path) -> None:
    r = CliRunner()
    base = ["--profile", "smoke", "--runs", str(tmp_path)]
    assert r.invoke(app, ["notes", *base]).exit_code == 0
    chk = r.invoke(app, ["kev-check", "--batch", "4", *base])
    assert chk.exit_code == 0, chk.output
    assert (tmp_path / "smoke" / "speed.json").exists() and "saved for `fns plan`" in chk.output
    pl = r.invoke(app, ["plan", "--cu-per-hour", "12", "--budget", "50", *base])
    assert pl.exit_code == 0 and "12 compute units per hour" in pl.output and "budget 50 CU" in pl.output
    paused = r.invoke(app, ["kev-rl", "--horizon", "1", "--max-minutes", "0.00001", *base])
    assert paused.exit_code == 0 and "paused" in paused.output
    st = r.invoke(app, ["status", *base])
    assert "checkpoint at update" in st.output
    done = r.invoke(app, ["kev-rl", "--horizon", "1", *base])
    assert "finished" in done.output
    assert r.invoke(app, ["ppo-all", *base]).exit_code == 0


def test_sessions_pay_setup_each_time() -> None:
    assert sessions_needed(2.0, 1.0, 0.15) == 3  # 0.85 h of work per session
    assert sessions_needed(1.0, 0.1, 0.15) is None
    cfg = full()
    text = render(cfg, Speed(**GUESS), 15, 76, session_cu=15)  # type: ignore[arg-type]
    assert "In sessions of 15 CU" in text and "MAX_MINUTES to 51" in text
    assert "too short" in render(cfg, Speed(**GUESS), 15, 76, session_cu=1)  # type: ignore[arg-type]
