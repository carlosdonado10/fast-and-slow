from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fast_and_slow.cli import app
from fast_and_slow.config import StudyConfig
from fast_and_slow.export.report import ReportBuilder, _clean
from fast_and_slow.notes.bank import NoteBank, split_cells
from fast_and_slow.study import Study


def test_note_bank_split_and_round_trip(tmp_path: Path, cfg: StudyConfig, bank: NoteBank) -> None:
    for cell, splits in bank.cells.items():
        assert splits["test"] and splits["train"], cell
        assert not set(splits["test"]) & set(splits["train"]), cell
    p = tmp_path / "bank.json"
    bank.save(p)
    assert NoteBank.load(p).cells == bank.cells
    one = split_cells({"x": ["only one", "only one", " "]}, cfg.notes)
    assert one["x"]["test"] == ["only one"] and one["x"]["train"] == ["only one"]


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    runs = tmp_path_factory.mktemp("runs")
    out = runs / "report.json"
    res = CliRunner().invoke(app, ["all", "--profile", "smoke", "--runs", str(runs), "--out", str(out)])
    assert res.exit_code == 0, res.output
    return runs, out


def test_cli_all_writes_every_stage(smoke_runs: tuple[Path, Path]) -> None:
    runs, _ = smoke_runs
    root = runs / "smoke"
    assert (root / "notes" / "bank.json").exists()
    assert (root / "refs" / "H1.json").exists()
    for arm in ("kev_zero", "kev_icl", "kev_rl", "ppo_num", "ppo_sig", "bayes", "random"):
        assert (root / "evals" / arm / "H4" / "seed0.json").exists(), arm
    assert (root / "curves" / "kev_rl" / "H4" / "seed0.json").exists()


def test_report_has_every_section_the_frontend_reads(smoke_runs: tuple[Path, Path]) -> None:
    _, out = smoke_runs
    r = json.loads(out.read_text())
    for key in (
        "meta",
        "arms",
        "references",
        "results",
        "curves",
        "efficiency",
        "horizon_dial",
        "costs",
        "replays",
        "desk",
        "hypotheses",
        "features",
        "decisions",
        "limitations",
    ):
        assert key in r, key
    assert r["meta"]["profile"] == "smoke"
    assert {h["id"] for h in r["hypotheses"]} == {"H1", "H2", "H3", "H4", "H5"}
    assert all(h["verdict"] in {"supported", "not supported", "inconclusive", "pending"} for h in r["hypotheses"])
    res = r["results"]["4"]["bayes"]
    assert res["pct_optimal"]["mean"] == pytest.approx(100)
    assert r["results"]["4"]["random"]["pct_optimal"]["mean"] == pytest.approx(0, abs=1e-6)
    assert set(r["efficiency"]["4"]["kev_rl"]["to_threshold"]) == {"80", "90", "95"}
    assert r["desk"]["weeks"][0]["state"].startswith("You are the replenishment planner")
    assert r["replays"] and r["replays"][0]["weeks"][0]["decisions"]


def test_stages_resume_without_redoing_work(smoke_runs: tuple[Path, Path]) -> None:
    runs, _ = smoke_runs
    res = CliRunner().invoke(app, ["refs", "--profile", "smoke", "--runs", str(runs)])
    assert res.exit_code == 0
    for cmd in (["notes"], ["kev"], ["kev-rl", "--horizon", "1"], ["ppo", "--variant", "num", "--horizon", "1"]):
        assert CliRunner().invoke(app, [*cmd, "--profile", "smoke", "--runs", str(runs)]).exit_code == 0
    exp = CliRunner().invoke(app, ["export", "--profile", "smoke", "--runs", str(runs), "--out", str(runs / "r2.json")])
    assert exp.exit_code == 0 and (runs / "r2.json").exists()


def test_kev_static_rejects_unknown_arms(tmp_path: Path, cfg: StudyConfig) -> None:
    with pytest.raises(ValueError):
        Study(cfg, tmp_path).kev_static("kev_magic")


def test_clean_makes_json_safe() -> None:
    import numpy as np

    out = _clean({"a": float("inf"), "b": [np.float64(1.5), np.int64(2)], 3: (float("nan"),)})
    assert out == {"a": None, "b": [1.5, 2], "3": [None]}


def _builder_with(results: dict, eff: dict, horizons: list[int], cfg: StudyConfig, tmp_path: Path) -> list[dict]:  # type: ignore[type-arg]
    cfg = cfg.model_copy(deep=True)
    cfg.eval.horizons = tuple(horizons)
    b = ReportBuilder(Study(cfg, tmp_path))
    return b.verdicts(results, eff)  # type: ignore[no-any-return]


def _res(pct: float, dec: float = 0.9, jud: float = 0.2, ece: float = 0.05) -> dict:  # type: ignore[type-arg]
    return {
        "pct_optimal": {"mean": pct},
        "calibration": {"ece": ece},
        "handoff": {"decidable": {"automation": {"0.05": dec}}, "judgement": {"automation": {"0.05": jud}}},
    }


def test_verdict_rules(cfg: StudyConfig, tmp_path: Path) -> None:
    H = [1, 12]
    results = {h: {"kev_zero": _res(90 - (20 if h == 12 else 0)), "kev_icl": _res(88), "kev_rl": _res(95)} for h in H}
    eff = {
        h: {
            "kev_zero": {"start": 85.0},
            "ppo_num": {"start": 0.0, "gain_per_decade": 20.0},
            "kev_rl": {"gain_per_decade": 3.0},
            "_crossover": {"ppo_num>kev_rl": 5000.0},
        }
        for h in H
    }
    v = {x["id"]: x["verdict"] for x in _builder_with(results, eff, H, cfg, tmp_path)}
    assert v == {"H1": "supported", "H2": "supported", "H3": "supported", "H4": "supported", "H5": "supported"}

    results = {
        h: {
            "kev_zero": _res(90, dec=0.1, jud=0.3),
            "kev_icl": _res(85, dec=0.1, jud=0.3),
            "kev_rl": _res(80, dec=0.1, jud=0.3),
        }
        for h in H
    }
    eff = {
        h: {
            "kev_zero": {"start": -5.0},
            "ppo_num": {"start": 0.0, "gain_per_decade": 1.0},
            "kev_rl": {"gain_per_decade": 3.0},
            "_crossover": {},
        }
        for h in H
    }
    v = {x["id"]: x["verdict"] for x in _builder_with(results, eff, H, cfg, tmp_path)}
    assert v == {
        "H1": "not supported",
        "H2": "not supported",
        "H3": "not supported",
        "H4": "not supported",
        "H5": "not supported",
    }

    v = {x["id"]: x["verdict"] for x in _builder_with({1: {}}, {1: {}}, [1], cfg, tmp_path)}
    assert set(v.values()) == {"pending"}


def test_status_and_kev_check(smoke_runs: tuple[Path, Path]) -> None:
    runs, _ = smoke_runs
    st = CliRunner().invoke(app, ["status", "--profile", "smoke", "--runs", str(runs)])
    assert st.exit_code == 0 and "stages complete" in st.output
    assert (
        st.output.strip().splitlines()[-1].split()[0].split("/")[0]
        == st.output.strip().splitlines()[-1].split()[0].split("/")[1]
    )
    ck = CliRunner().invoke(app, ["kev-check", "--profile", "smoke", "--runs", str(runs)])
    assert ck.exit_code == 0, ck.output
    assert "RL can run" in ck.output
