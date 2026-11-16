"""Typed configuration for the whole study.

Two profiles ship with the package:
- ``smoke``: runs end to end on a laptop in minutes (fake Kev, template notes, tiny budgets).
- ``full``: the Colab H100 study described in docs/spec.md (~25 H100-hours).

Every number that shapes the world or the budget lives here, so the report can print the exact config it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class WorldConfig(BaseModel):
    """The planner's replenishment desk (docs/spec.md section 2)."""

    regimes: tuple[str, ...] = ("low", "normal", "surge")
    demand_means: tuple[float, ...] = (3.0, 10.0, 24.0)
    regime_prior: tuple[float, ...] = (0.25, 0.5, 0.25)
    stay_prob: float = 0.5  # regimes shift often, so last week's sales are a weak guide and the inbox matters
    max_demand: int = 40
    order_menu: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 40)
    inventory_cap: int = 60
    start_inventory: int = 0
    price: float = 14.0
    unit_cost: float = 5.0
    holding_cost: float = 4.0
    stockout_penalty: float = 10.0
    fixed_order_cost: float = 10.0
    salvage_value: float = 0.0
    # signal model
    note_prob: float = 0.9
    distractor_prob: float = 0.15
    sources: tuple[str, ...] = ("rumour", "sales_rep", "customer_order")
    source_accuracy: tuple[float, ...] = (0.65, 0.80, 0.95)
    source_mix: tuple[float, ...] = (0.4, 0.4, 0.2)


class OracleConfig(BaseModel):
    belief_resolution: int = 20  # simplex grid step = 1/resolution
    acceptable_regret_usd: float = 3.0  # an order is "acceptable" if it gives up at most $3 of expected profit
    decidable_posterior_floor: float = 0.15


class NotesConfig(BaseModel):
    writer: Literal["templates", "llm"] = "templates"
    llm_model: str = "microsoft/Phi-4-mini-instruct"
    per_cell: int = 60
    test_fraction: float = 1 / 3
    seed: int = 7


class KevConfig(BaseModel):
    run: str = "jaredpalmer/kev-4b@v1.0"
    fake: bool = False  # smoke profile: a deterministic stand-in with the same interface
    dtype: Literal["bf16", "fp32"] = "bf16"
    icl_examples: int = 4
    # RL (PPO on the pointer-head policy). Budgeted for ~100 Colab compute units: run `fns plan` after `fns kev-check`.
    rl_horizons: tuple[int, ...] = (
        1,
        12,
    )  # the two ends of the dial; zero-shot / in-context still run at every horizon
    rl_updates: int = 60
    rl_group_size: int = 8  # rollouts per scenario seed (shared exogenous randomness)
    rl_groups_per_update: int = 4
    rl_epochs: int = 2
    rl_minibatch: int = 32
    rl_lr: float = 1e-5
    rl_head_lr: float = 1e-4
    rl_clip: float = 0.2
    rl_entropy: float = 0.01
    rl_reward_scale: float = 0.01
    eval_every_updates: int = 6
    checkpoint_every_updates: int = 3  # a few minutes of GPU time between checkpoints


class ClassicRLConfig(BaseModel):
    variants: tuple[str, ...] = ("num", "emb", "sig")
    seeds: tuple[int, ...] = (0, 1, 2)  # PPO runs on a laptop CPU for free, so it gets more seeds than Kev
    total_decisions: int = 300_000
    n_envs: int = 8
    eval_points: int = 12  # log-spaced checkpoints along training
    embedder: str = "BAAI/bge-small-en-v1.5"
    checkpoint_every_decisions: int = 25_000
    embed_dim: int = 384


class EvalConfig(BaseModel):
    horizons: tuple[int, ...] = (1, 4, 12)
    scenarios_per_horizon: int = 200
    curve_scenarios: int = 64  # held-out subset scored at every learning-curve checkpoint (the final eval uses all)
    eval_seed: int = 10_000  # held-out scenario seeds start here; training seeds stay below
    bootstrap: int = 1000
    error_budget: float = 0.05
    efficiency_thresholds: tuple[float, ...] = (80.0, 90.0, 95.0)  # % of optimal, for decisions-to-threshold
    replay_scenarios: int = 3
    h100_usd_per_hour: float = 3.0
    # Colab bills compute units (CU). Check the rate for your GPU in Colab's resources panel; ~15/h is an H100 guess.
    cu_per_hour: float = 15.0
    budget_cu: float = 100.0


class ShiftConfig(BaseModel):
    """H6: regime changes over a deployment (docs/spec.md section 2b)."""

    shifts: tuple[str, ...] = ("contract_penalty", "competitor_entry")
    horizon: int = 4  # monthly plans
    weeks: int = 52
    shift_week: int = 20  # must sit on a plan boundary
    rebuild_weeks: int = 6  # time to update the simulator before offline retraining
    live_units: int = 50  # products / sites sharing one policy: live decisions per week
    ppo_variants: tuple[str, ...] = ("num", "emb")
    retrain_decisions: int = 200_000
    retrain_eval_points: int = 10
    scenarios: int = 120  # held-out plans in the new world
    detect_streams: int = 30  # 52-week streams for the alarm
    calibration_streams: int = 30  # no-shift streams to set each detector's threshold
    false_alarm_target: float = 0.02  # alarms per pre-shift week both detectors are calibrated to
    recover_at: float = 90.0  # % of optimal that counts as recovered


class StudyConfig(BaseModel):
    profile: str = "smoke"
    seeds: tuple[int, ...] = (0,)  # Kev RL seeds (GPU, the expensive arm)
    runs_dir: Path = Path("runs")
    world: WorldConfig = Field(default_factory=WorldConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    kev: KevConfig = Field(default_factory=KevConfig)
    classic: ClassicRLConfig = Field(default_factory=ClassicRLConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    shift: ShiftConfig = Field(default_factory=ShiftConfig)


def smoke() -> StudyConfig:
    return StudyConfig(
        profile="smoke",
        seeds=(0,),
        notes=NotesConfig(writer="templates", per_cell=12),
        kev=KevConfig(
            fake=True,
            rl_horizons=(1, 4),
            rl_updates=4,
            rl_group_size=4,
            rl_groups_per_update=2,
            rl_minibatch=8,
            eval_every_updates=2,
        ),
        classic=ClassicRLConfig(total_decisions=6_000, n_envs=4, eval_points=4, variants=("num", "sig"), seeds=(0,)),
        eval=EvalConfig(
            horizons=(1, 4), scenarios_per_horizon=24, curve_scenarios=12, bootstrap=200, replay_scenarios=2
        ),
        shift=ShiftConfig(
            weeks=24,
            shift_week=8,
            rebuild_weeks=4,
            live_units=20,
            ppo_variants=("num",),
            retrain_decisions=3_000,
            retrain_eval_points=3,
            scenarios=8,
            detect_streams=4,
            calibration_streams=4,
        ),
    )


def full() -> StudyConfig:
    return StudyConfig(profile="full", seeds=(0,), notes=NotesConfig(writer="llm"))


PROFILES = {"smoke": smoke, "full": full}


def load(profile_or_path: str) -> StudyConfig:
    """A profile name, or a YAML file whose keys override the profile named in its `profile` key."""
    if profile_or_path in PROFILES:
        return PROFILES[profile_or_path]()
    data = yaml.safe_load(Path(profile_or_path).read_text())
    base = PROFILES[data.pop("profile", "smoke")]().model_dump()
    _deep_update(base, data)
    return StudyConfig.model_validate(base)


def _deep_update(base: dict[str, Any], over: dict[str, Any]) -> None:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
