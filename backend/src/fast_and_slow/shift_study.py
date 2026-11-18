"""H6 stages: measure everything the transition timelines need, for one shift. Resumable like the rest of the study.

runs/<profile>/shifts/<shift>/
  refs.json          random / Bayes / perfect-information returns on the new world
  ppo_<variant>.json warm-started retraining curve on the new world (+ the old-world score it starts from)
  kev.json           Kev on the new world: told about the change (declared brief) and not told (old brief)
  detect.json        weekly alarm scores, Kev vs Page-Hinkley, each calibrated to the same false-alarm rate
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agents.base import Agent
from .agents.baselines import BaseStockAgent, BayesAgent, OracleAgent, RandomAgent
from .agents.kev_agent import KevAgent
from .eval import detect as D
from .eval.metrics import pct_of_optimal
from .eval.runner import Labeller, eval_scenarios, evaluate
from .progress import Progress
from .sim.render import CHANGE_Q, change_question, request
from .sim.shifts import get
from .study import Study

NEW_WORLD_SEEDS = 60_000
CALIBRATION_SEEDS = 70_000
STREAM_SEEDS = 80_000


class ShiftStudy:
    def __init__(self, study: Study, name: str):
        self.s, self.shift = study, get(name)
        self.c = study.cfg.shift
        self.old = study.cfg.world
        self.new = self.shift.world(self.old)
        self.H = self.c.horizon
        self.dir = study.root / "shifts" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lab: Labeller | None = None

    # ------------------------------------------------------------ helpers
    @property
    def labeller(self) -> Labeller:
        if self._lab is None:
            self._lab = Labeller(self.new, self.H, self.s.cfg.oracle)
        return self._lab

    def scenarios(self) -> list[Any]:
        return eval_scenarios(self.new, self.H, self.c.scenarios, NEW_WORLD_SEEDS)

    def _write(self, name: str, data: Any) -> None:
        (self.dir / name).write_text(json.dumps(data, default=float))

    def _read(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text())

    def scorer(self) -> Callable[..., float]:
        refs = self.refs()

        def score(agent: Agent, label: str | None = None) -> float:
            r = evaluate(agent, self.new, self.s.bank, self.scenarios(), self.labeller, record=False, label=label)
            return pct_of_optimal(float(np.mean(r.returns)), refs["random"], refs["bayes"])

        return score

    # ------------------------------------------------------------ stages
    def refs(self, force: bool = False) -> dict[str, float]:
        p = self.dir / "refs.json"
        if not p.exists() or force:
            arms: list[Agent] = [
                RandomAgent(self.new),
                BayesAgent(self.new, self.H, self.s.cfg.oracle),
                OracleAgent(self.new, self.H),
            ]
            mean = {
                a.name: float(
                    np.mean(
                        evaluate(
                            a,
                            self.new,
                            self.s.bank,
                            self.scenarios(),
                            self.labeller,
                            record=False,
                            label=f"shift {self.shift.name} refs {a.name}",
                        ).returns
                    )
                )
                for a in arms
            }
            self._write("refs.json", {"mean_return": mean, "world": self.new.model_dump()})
        return dict(self._read("refs.json")["mean_return"])

    def ppo(self, variant: str, force: bool = False) -> None:
        """Retrain the deployed PPO (trained on the old world) on the new one, from its old weights."""
        from .agents.classic_rl import train_ppo

        out = f"ppo_{variant}.json"
        if (self.dir / out).exists() and not force:
            return
        self.s.ppo(variant, self.H, 0)  # the deployed policy: old world, same horizon (skipped if already trained)
        init = self.s.root / "checkpoints" / f"ppo_{variant}" / f"H{self.H}" / "seed0" / "model.zip"
        old_eval = self.s.load_eval(f"ppo_{variant}", self.H, 0)
        assert old_eval is not None
        refs_old = self.s.refs(self.H)
        pre = pct_of_optimal(float(np.mean(old_eval.returns)), refs_old["random"], refs_old["bayes"])
        cls = self.s.cfg.classic
        emb = self.s.embedder() if variant == "emb" else None
        _, curve = train_ppo(
            self.new,
            self.s.bank,
            self.H,
            variant,
            emb,
            self.c.retrain_decisions,
            cls.n_envs,
            self.c.retrain_eval_points,
            self.scorer(),
            seed=0,
            checkpoint=self.dir / f"ckpt_ppo_{variant}",
            save_every=cls.checkpoint_every_decisions,
            init=init,
            label=f"shift {self.shift.name} ppo_{variant} retrain",
        )
        self._write(out, {"pre_shift_score": pre, "points": curve})

    def kev(self, force: bool = False) -> None:
        if (self.dir / "kev.json").exists() and not force:
            return
        pol = self.s.policy()
        score = self.scorer()
        tag = f"shift {self.shift.name} kev"
        told = score(
            KevAgent("kev_declared", self.new, pol, context=self.shift.declared_note, ask_regime=False),
            f"{tag} (told about the change)",
        )
        not_told = score(KevAgent("kev_stale_brief", self.old, pol, ask_regime=False), f"{tag} (not told)")
        self._write("kev.json", {"declared": told, "stale_brief": not_told})

    def detect(self, force: bool = False) -> None:
        if (self.dir / "detect.json").exists() and not force:
            return
        tool = BaseStockAgent.tune(self.old, self.H, list(range(200)))
        bank, c = self.s.bank, self.c

        def streams(n: int, first: int, shifted: bool) -> list[D.Stream]:
            return [
                D.make_stream(
                    self.old,
                    self.new,
                    self.shift if shifted else None,
                    bank,
                    c.weeks,
                    c.shift_week if shifted else None,
                    first + i,
                    tool,
                )
                for i in range(n)
            ]

        calib = streams(c.calibration_streams, CALIBRATION_SEEDS, False)
        test = streams(c.detect_streams, STREAM_SEEDS, True)
        out: dict[str, Any] = {"weeks": c.weeks, "shift_week": c.shift_week}
        for name, fn in (("kev", self._kev_scores), ("page_hinkley", self._ph_scores)):
            cal_scores, test_scores = fn(calib, "calibration years"), fn(test, "test years")
            thr = D.calibrate(cal_scores, c.false_alarm_target)
            out[name] = {
                "matched": D.summarise(test_scores, test, thr),
                "mean_score_by_week": np.mean(np.asarray(test_scores), axis=0).round(4).tolist(),
            }
            if name == "kev":
                out[name]["at_0.5"] = D.summarise(test_scores, test, 0.5)
        self._write("detect.json", out)

    def _kev_scores(self, streams: list[D.Stream], what: str = "") -> list[list[float]]:
        pol = self.s.policy()
        reqs = []
        for st in streams:
            for o in st.observations:
                r = request(self.old, o)  # the old brief: nobody has told it anything
                r["questions"] = {CHANGE_Q: change_question()}
                reqs.append(r)
        probs: list[float] = []
        prog = Progress(f"shift {self.shift.name} kev alarm, {what}".rstrip(", "), len(reqs), "weeks")
        for i in range(0, len(reqs), 32):
            prog.update(i)
            for z in pol.score(reqs[i : i + 32]):
                probs.append(float(torch.softmax(z[CHANGE_Q].float(), -1)[1]))
        prog.update(len(reqs))
        weeks = self.c.weeks
        return [probs[k * weeks : (k + 1) * weeks] for k in range(len(streams))]

    def _ph_scores(self, streams: list[D.Stream], what: str = "") -> list[list[float]]:
        return [D.page_hinkley(self.old, st) for st in streams]

    # ------------------------------------------------------------ for the export
    def results(self) -> dict[str, Any] | None:
        need = ["refs.json", "kev.json", "detect.json"] + [f"ppo_{v}.json" for v in self.c.ppo_variants]
        if not all((self.dir / n).exists() for n in need):
            return None
        return {
            "refs": self._read("refs.json")["mean_return"],
            "kev": self._read("kev.json"),
            "detect": self._read("detect.json"),
            "ppo": {v: self._read(f"ppo_{v}.json") for v in self.c.ppo_variants},
        }


def run_shift(study: Study, name: str, part: str, force: bool = False) -> Path:
    """part: 'cpu' (references + PPO retraining; laptop) | 'gpu' (Kev on the new world + the weekly alarm)."""
    ss = ShiftStudy(study, name)
    if part == "cpu":
        ss.refs(force)
        for v in ss.c.ppo_variants:
            ss.ppo(v, force)
    elif part == "gpu":
        ss.refs(force)
        ss.kev(force)
        ss.detect(force)
    else:
        raise ValueError(f"part must be cpu or gpu, not {part!r}")
    return ss.dir
