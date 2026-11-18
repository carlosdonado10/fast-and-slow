"""The study's stages. Each one is resumable: it writes its outputs under runs/<profile>/ and skips work already done
(pass force=True to redo). The CLI and the Colab notebook are thin wrappers around these functions.

runs/<profile>/
  config.json                       the exact StudyConfig
  notes/bank.json                   the frozen note bank
  refs/H{h}.json                    reference returns + full decision records for random, basestock, bayes, oracle
  evals/<arm>/H{h}/seed{s}.json     final evaluation (all held-out scenarios, every decision labelled)
  curves/<arm>/H{h}/seed{s}.json    learning curve: [{decisions, score, seconds}] (+ RL update logs)
  weights/<arm>/H{h}/seed{s}/       trained Kev adapters (not committed; sync to Drive)
  checkpoints/<arm>/H{h}/seed{s}/   mid-run state (policy, optimizer, RNG, curve) so a dropped Colab session resumes
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .agents.base import Agent
from .agents.baselines import BaseStockAgent, BayesAgent, OracleAgent, RandomAgent
from .agents.features import CachedEmbedder, Embedder, HashEmbedder
from .agents.kev_agent import KevAgent, icl_context
from .agents.kev_rl import KevRLTrainer
from .agents.policy import DecisionPolicy, load_policy
from .config import StudyConfig
from .eval.metrics import pct_of_optimal
from .eval.runner import EvalResult, Labeller, eval_scenarios, evaluate
from .notes.bank import NoteBank

log = logging.getLogger("fast_and_slow")
REFERENCE_ARMS = ("random", "basestock", "bayes", "oracle")


class Study:
    def __init__(self, cfg: StudyConfig, root: Path | None = None):
        self.cfg = cfg
        self.root = (root or cfg.runs_dir) / cfg.profile
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(cfg.model_dump_json(indent=1))
        self._labellers: dict[int, Labeller] = {}
        self._bank: NoteBank | None = None
        self._policy: DecisionPolicy | None = None

    # ------------------------------------------------------------ helpers
    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def labeller(self, h: int) -> Labeller:
        if h not in self._labellers:
            self._labellers[h] = Labeller(self.cfg.world, h, self.cfg.oracle)
        return self._labellers[h]

    @property
    def bank(self) -> NoteBank:
        if self._bank is None:
            p = self.path("notes", "bank.json")
            if not p.exists():
                self.notes()
            self._bank = NoteBank.load(p)
        return self._bank

    def scenarios(self, h: int, n: int | None = None) -> list[Any]:
        e = self.cfg.eval
        return eval_scenarios(self.cfg.world, h, n or e.scenarios_per_horizon, e.eval_seed)

    def refs(self, h: int) -> dict[str, float]:
        p = self.path("refs", f"H{h}.json")
        if not p.exists():
            self.references()
        return dict(json.loads(p.read_text())["mean_return"])

    def curve_scorer(self, h: int) -> Callable[[Agent], float]:
        """% of optimal on the curve subset, against references computed on that same subset."""
        sc = self.scenarios(h, self.cfg.eval.curve_scenarios)
        lab = self.labeller(h)
        w = self.cfg.world
        rnd = float(np.mean(evaluate(RandomAgent(w), w, self.bank, sc, lab, record=False).returns))
        bay = float(np.mean(evaluate(BayesAgent(w, h, self.cfg.oracle), w, self.bank, sc, lab, record=False).returns))

        def score(agent: Agent) -> float:
            r = evaluate(agent, w, self.bank, sc, lab, record=False)
            return pct_of_optimal(float(np.mean(r.returns)), rnd, bay)

        return score

    def policy(self, fresh: bool = False) -> DecisionPolicy:
        k = self.cfg.kev
        if fresh or self._policy is None:
            self._policy = load_policy(k.run, k.fake, k.dtype)
        return self._policy

    def embedder(self) -> Embedder:
        if self.cfg.kev.fake:
            return CachedEmbedder(HashEmbedder(64))
        from .agents.embedder_gpu import SentenceEmbedder  # pragma: no cover - gpu extra

        return CachedEmbedder(SentenceEmbedder(self.cfg.classic.embedder))  # pragma: no cover

    @staticmethod
    def _write(p: Path, data: Any) -> None:
        p.write_text(json.dumps(data, default=float))

    # ------------------------------------------------------------ stages
    def notes(self, force: bool = False) -> Path:
        p = self.path("notes", "bank.json")
        if p.exists() and not force:
            return p
        if self.cfg.notes.writer == "llm":
            from .notes import llm_writer  # pragma: no cover - gpu extra

            bank = llm_writer.write(self.cfg.world, self.cfg.notes)  # pragma: no cover
        else:
            from .notes import templates

            bank = templates.write(self.cfg.world, self.cfg.notes)
        bank.save(p)
        self._bank = bank
        return p

    def references(self, force: bool = False) -> None:
        w = self.cfg.world
        for h in self.cfg.eval.horizons:
            p = self.path("refs", f"H{h}.json")
            if p.exists() and not force:
                continue
            sc, lab = self.scenarios(h), self.labeller(h)
            arms: list[Agent] = [
                RandomAgent(w),
                BaseStockAgent.tune(w, h, list(range(200))),
                BayesAgent(w, h, self.cfg.oracle),
                OracleAgent(w, h),
            ]
            results = {a.name: evaluate(a, w, self.bank, sc, lab, label=f"refs H{h} {a.name}") for a in arms}
            bs = arms[1]
            assert isinstance(bs, BaseStockAgent)
            self._write(
                p,
                {
                    "horizon": h,
                    "mean_return": {k: float(np.mean(r.returns)) for k, r in results.items()},
                    "basestock_params": {"s": bs.s, "S": bs.S},
                    "bayes_grid_value": lab.bayes.value(),
                },
            )
            for name, r in results.items():
                self._write(self.path("evals", name, f"H{h}", "seed0.json"), r.to_json())

    def kev_static(self, arm: str, force: bool = False) -> None:
        """kev_zero or kev_icl: no training; one curve point at x = examples shown (0 or k) and a final evaluation."""
        if arm not in ("kev_zero", "kev_icl"):
            raise ValueError(arm)
        w, k = self.cfg.world, self.cfg.kev
        for h in self.cfg.eval.horizons:
            p = self.path("evals", arm, f"H{h}", "seed0.json")
            if p.exists() and not force:
                continue
            ctx = icl_context(w, self.bank, h, k.icl_examples) if arm == "kev_icl" else None
            agent = KevAgent(arm, w, self.policy(), context=ctx)
            t0 = time.perf_counter()
            res = evaluate(agent, w, self.bank, self.scenarios(h), self.labeller(h), label=f"{arm} H{h}")
            seconds = time.perf_counter() - t0
            self._write(p, res.to_json())
            refs = self.refs(h)
            score = pct_of_optimal(float(np.mean(res.returns)), refs["random"], refs["bayes"])
            shown = k.icl_examples if arm == "kev_icl" else 0
            self._write(
                self.path("curves", arm, f"H{h}", "seed0.json"),
                {
                    "points": [{"decisions": shown, "score": score, "seconds": 0.0}],
                    "eval_seconds": seconds,
                    "context": ctx,
                },
            )

    def kev_rl(self, h: int, seed: int, force: bool = False, max_minutes: float | None = None) -> bool:
        """Returns True when the run is finished, False when it paused at a checkpoint (time budget reached)."""
        p = self.path("evals", "kev_rl", f"H{h}", f"seed{seed}.json")
        if p.exists() and not force:
            return True
        w = self.cfg.world
        name = f"kev_rl H{h} seed{seed}"
        log.info("%s: loading Kev and scoring the reference arms for the learning curve", name)
        policy = self.policy(fresh=True)  # every run starts from the released weights
        scorer = self.curve_scorer(h)
        curve_path = self.path("curves", "kev_rl", f"H{h}", f"seed{seed}.json")

        def evaluate_policy(pol: DecisionPolicy) -> float:
            return scorer(KevAgent("kev_rl", w, pol, ask_regime=False))

        trainer = KevRLTrainer(
            w, self.bank, h, policy, self.cfg.kev, seed=seed, train_seed_range=self.cfg.eval.eval_seed
        )
        rl_log = trainer.train(
            evaluate_policy,
            on_point=lambda lg: self._write(curve_path, {"points": lg.points, "updates": lg.updates}),
            checkpoint=self.root / "checkpoints" / "kev_rl" / f"H{h}" / f"seed{seed}",
            deadline=time.monotonic() + max_minutes * 60 if max_minutes else None,
            label=name,
        )
        if rl_log.paused:
            log.info(
                "kev_rl H%s seed%s paused at %s decisions; run the same command to continue", h, seed, trainer.decisions
            )
            self._policy = None
            return False
        self._write(curve_path, {"points": rl_log.points, "updates": rl_log.updates, "train_seconds": trainer.seconds})
        policy.save(self.root / "weights" / "kev_rl" / f"H{h}" / f"seed{seed}")
        res = evaluate(
            KevAgent("kev_rl", w, policy), w, self.bank, self.scenarios(h), self.labeller(h), label=f"{name} final eval"
        )
        self._write(p, res.to_json())
        self._policy = None  # the next arm reloads clean weights
        return True

    def ppo(self, variant: str, h: int, seed: int, force: bool = False) -> None:
        from .agents.classic_rl import PPOAgent, train_ppo

        arm = f"ppo_{variant}"
        p = self.path("evals", arm, f"H{h}", f"seed{seed}.json")
        if p.exists() and not force:
            return
        w, c = self.cfg.world, self.cfg.classic
        emb = self.embedder() if variant == "emb" else None
        model, curve = train_ppo(
            w,
            self.bank,
            h,
            variant,
            emb,
            c.total_decisions,
            c.n_envs,
            c.eval_points,
            self.curve_scorer(h),
            seed=seed,
            checkpoint=self.root / "checkpoints" / arm / f"H{h}" / f"seed{seed}",
            save_every=c.checkpoint_every_decisions,
            label=f"{arm} H{h} seed{seed}",
        )
        self._write(self.path("curves", arm, f"H{h}", f"seed{seed}.json"), {"points": curve})
        res = evaluate(
            PPOAgent(arm, w, model, variant, emb),
            w,
            self.bank,
            self.scenarios(h),
            self.labeller(h),
            label=f"{arm} H{h} seed{seed} final eval",
        )
        self._write(p, res.to_json())

    # ------------------------------------------------------------ loading results
    def load_eval(self, arm: str, h: int, seed: int = 0) -> EvalResult | None:
        p = self.root / "evals" / arm / f"H{h}" / f"seed{seed}.json"
        return EvalResult.from_json(json.loads(p.read_text())) if p.exists() else None

    def arms_present(self) -> list[str]:
        d = self.root / "evals"
        return sorted(x.name for x in d.iterdir()) if d.exists() else []

    # ------------------------------------------------------------ the run matrix
    def kev_rl_runs(self) -> list[tuple[int, int]]:
        return [(h, s) for s in self.cfg.seeds for h in self.cfg.kev.rl_horizons]

    def ppo_runs(self) -> list[tuple[str, int, int]]:
        c = self.cfg.classic
        return [(v, h, s) for s in c.seeds for h in self.cfg.eval.horizons for v in c.variants]
