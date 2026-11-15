"""runs/<profile>/ -> report.json, the only thing the React report reads."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.base import Agent
from ..eval import metrics as M
from ..eval.runner import EvalResult
from ..sim.desk import Episode
from ..sim.world import CLAIMS, Note
from ..study import Study
from . import narrative

SYSTEM_ONE = ("kev_zero", "kev_icl", "kev_rl")
CLASSIC = ("ppo_num", "ppo_emb", "ppo_sig")


def _finite(x: float | None) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _clean(o: Any) -> Any:
    """JSON-safe: inf/nan -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _finite(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return o


class ReportBuilder:
    def __init__(self, study: Study):
        self.s = study
        self.cfg = study.cfg
        self.H = list(self.cfg.eval.horizons)

    # ------------------------------------------------------------ loading
    def evals(self, arm: str, h: int) -> list[EvalResult]:
        out = []
        for seed in range(16):
            r = self.s.load_eval(arm, h, seed)
            if r is not None:
                out.append(r)
        return out

    def curves(self, arm: str, h: int) -> list[list[M.CurvePoint]]:
        d = self.s.root / "curves" / arm / f"H{h}"
        if not d.exists():
            return []
        out = []
        for f in sorted(d.glob("seed*.json")):
            pts = json.loads(f.read_text())["points"]
            out.append(
                [M.CurvePoint(int(p["decisions"]), float(p["score"]), float(p.get("seconds", 0.0))) for p in pts]
            )
        return out

    @staticmethod
    def mean_curve(curves: list[list[M.CurvePoint]]) -> list[M.CurvePoint]:
        if not curves:
            return []
        n = min(len(c) for c in curves)
        return [
            M.CurvePoint(
                int(np.mean([c[i].decisions for c in curves])),
                float(np.mean([c[i].score for c in curves])),
                float(np.mean([c[i].seconds for c in curves])),
            )
            for i in range(n)
        ]

    # ------------------------------------------------------------ sections
    def results(self, h: int, refs: dict[str, float]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        budget = self.cfg.eval.error_budget
        for arm in self.s.arms_present():
            runs = self.evals(arm, h)
            if not runs:
                continue
            per_seed = [float(np.mean(r.returns)) for r in runs]
            returns = [x for r in runs for x in r.returns]
            recs = [d for r in runs for d in r.decisions]
            pct = [M.pct_of_optimal(x, refs["random"], refs["bayes"]) for x in returns]
            inbox = [M.inbox_value(x, refs["basestock"], refs["bayes"]) for x in returns]
            rel = M.reliability([d.confidence for d in recs], [d.acceptable for d in recs])
            lat = [d.latency_ms for d in recs]
            out[arm] = {
                "seeds": len(runs),
                "return_mean": float(np.mean(per_seed)),
                "pct_optimal": {"mean": float(np.mean(pct)), "ci": M.bootstrap_ci(pct, self.cfg.eval.bootstrap)},
                "inbox_value": {"mean": float(np.mean(inbox)), "ci": M.bootstrap_ci(inbox, self.cfg.eval.bootstrap)},
                "pct_of_perfect_info": M.pct_of_optimal(float(np.mean(returns)), refs["random"], refs["oracle"]),
                "acceptable_rate": float(np.mean([d.acceptable for d in recs])),
                "mean_regret": float(np.mean([d.regret for d in recs])),
                "calibration": {"ece": rel.ece, "brier": rel.brier, "bins": rel.bins},
                "regime": M.regime_calibration(recs),
                "handoff": M.handoff(recs, (0.01, 0.02, budget, 0.1)),
                "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
            }
        return out

    def learning(self, h: int) -> tuple[dict[str, Any], dict[str, Any]]:
        curves_out: dict[str, Any] = {}
        eff: dict[str, Any] = {}
        all_curves = {arm: self.curves(arm, h) for arm in (*SYSTEM_ONE, *CLASSIC)}
        all_curves = {a: c for a, c in all_curves.items() if c}
        if not all_curves:
            return curves_out, eff
        hi = max(max(p.decisions for p in c) for cs in all_curves.values() for c in cs)
        lo = 10.0
        for arm, cs in all_curves.items():
            mean = self.mean_curve(cs)
            curves_out[arm] = {
                "mean": [{"decisions": p.decisions, "score": p.score, "seconds": p.seconds} for p in mean],
                "seeds": [[{"decisions": p.decisions, "score": p.score} for p in c] for c in cs],
                "band": self._band(cs),
            }
            per_seed = [M.efficiency_summary(c, self.cfg.eval.efficiency_thresholds, lo, max(hi, lo * 10)) for c in cs]
            eff[arm] = {
                **M.efficiency_summary(mean, self.cfg.eval.efficiency_thresholds, lo, max(hi, lo * 10)),
                "per_seed": per_seed,
            }
        cross = {}
        for fast in SYSTEM_ONE:
            for slow in CLASSIC:
                if fast in all_curves and slow in all_curves:
                    cross[f"{slow}>{fast}"] = M.crossover(
                        self.mean_curve(all_curves[fast]), self.mean_curve(all_curves[slow])
                    )
        eff["_crossover"] = cross
        return curves_out, eff

    @staticmethod
    def _band(cs: list[list[M.CurvePoint]]) -> list[dict[str, float]]:
        n = min(len(c) for c in cs)
        return [
            {
                "decisions": float(np.mean([c[i].decisions for c in cs])),
                "lo": float(np.min([c[i].score for c in cs])),
                "hi": float(np.max([c[i].score for c in cs])),
            }
            for i in range(n)
        ]

    def replays(self) -> list[dict[str, Any]]:
        """A few held-out scenarios with every arm's decisions, week by week, for the side-by-side view (F7)."""
        out = []
        w, bank = self.cfg.world, self.s.bank
        h = max(self.H)
        scen = self.s.scenarios(h)
        arms = [
            a for a in ("bayes", "basestock", "kev_zero", "kev_icl", "kev_rl", "ppo_emb", "ppo_sig") if self.evals(a, h)
        ]
        by_arm = {a: {(d.seed, d.week): d for d in self.evals(a, h)[0].decisions} for a in arms}

        # prefer scenarios where Kev and PPO disagree most
        def disagreement(seed: int) -> int:
            ks = [by_arm[a] for a in arms if a.startswith("kev")]
            ps = [by_arm[a] for a in arms if a.startswith("ppo")]
            if not ks or not ps:
                return 0
            return sum(ks[0][(seed, t)].action != ps[0][(seed, t)].action for t in range(h) if (seed, t) in ks[0])

        ranked = sorted(scen, key=lambda sc: -disagreement(sc.seed))[: self.cfg.eval.replay_scenarios]
        for sc in ranked:
            texts = bank.texts(sc, "test")
            weeks = []
            for t in range(h):
                note: Note = sc.notes[t]
                weeks.append(
                    {
                        "week": t + 1,
                        "note": texts[t],
                        "note_source": w.sources[note.source] if note.kind == "signal" else note.kind,
                        "note_claim": CLAIMS[note.claim] if note.kind == "signal" else None,
                        "true_regime": w.regimes[sc.regimes[t]],
                        "demand": sc.demands[t],
                        "decisions": {
                            a: {
                                "order": w.order_menu[d.action],
                                "confidence": d.confidence,
                                "acceptable": d.acceptable,
                                "inventory": d.inventory,
                                "decidable": d.decidable,
                                "bayes_order": w.order_menu[d.bayes_action],
                                "posterior": d.bayes_posterior,
                                "reward": d.reward,
                            }
                            for a in arms
                            if (d := by_arm[a].get((sc.seed, t))) is not None
                        },
                    }
                )
            out.append(
                {
                    "horizon": h,
                    "seed": sc.seed,
                    "weeks": weeks,
                    "totals": {a: float(sum(by_arm[a][(sc.seed, t)].reward for t in range(h))) for a in arms},
                }
            )
        return out

    def desk_example(self) -> dict[str, Any]:
        """One short scenario for F3: what the planner sees each week, and the truth behind it."""
        w, bank = self.cfg.world, self.s.bank
        h = 4 if 4 in self.H else self.H[0]
        sc = self.s.scenarios(h)[0]
        ep = Episode(w, sc, bank.texts(sc, "test"))
        lab = self.s.labeller(h)
        from ..agents.baselines import BayesAgent, bayes_belief
        from ..sim.render import request

        agent: Agent = BayesAgent(w, h, self.cfg.oracle)
        agent.begin([sc])
        weeks = []
        while not ep.done:
            o = ep.observe()
            b = bayes_belief(lab.bayes, o)
            d = agent.act(o, 0)
            req = request(w, o)
            out = ep.step(w.order_menu[d.action])
            weeks.append(
                {
                    "week": o.week + 1,
                    "state": req["state"],
                    "questions": req["questions"],
                    "posterior": dict(zip(w.regimes, map(float, b), strict=True)),
                    "true_regime": w.regimes[sc.regimes[o.week]],
                    "bayes_order": out.order,
                    "demand": out.demand,
                    "profit": out.reward,
                }
            )
        return {"horizon": h, "weeks": weeks}

    def costs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        usd = self.cfg.eval.h100_usd_per_hour
        for arm in (*SYSTEM_ONE, *CLASSIC):
            secs, lat = [], []
            for h in self.H:
                d = self.s.root / "curves" / arm / f"H{h}"
                for f in sorted(d.glob("seed*.json")) if d.exists() else []:
                    data = json.loads(f.read_text())
                    secs.append(float(data.get("train_seconds", data["points"][-1].get("seconds", 0.0))))
                for r in self.evals(arm, h):
                    lat += [x.latency_ms for x in r.decisions]
            if not lat:
                continue
            gpu = arm.startswith("kev")
            train_h = sum(secs) / 3600
            p50 = float(np.percentile(lat, 50))
            out[arm] = {
                "hardware": "H100" if gpu and not self.cfg.kev.fake else "CPU",
                "train_hours": train_h,
                "train_usd": train_h * usd if gpu else 0.0,
                "ms_per_decision_p50": p50,
                "ms_per_decision_p95": float(np.percentile(lat, 95)),
                "usd_per_1k_decisions": (p50 / 1000 / 3600 * usd * 1000) if gpu else 0.0,
            }
        return out

    # ------------------------------------------------------------ verdicts
    def verdicts(self, results: dict[int, Any], eff: dict[int, Any]) -> list[dict[str, Any]]:
        hyp = {h["id"]: dict(h) for h in narrative.HYPOTHESES}

        def pct(h: int, arm: str) -> float | None:
            r = results.get(h, {}).get(arm)
            return r["pct_optimal"]["mean"] if r else None

        def put(i: str, verdict: str, evidence: str) -> None:
            hyp[i]["verdict"], hyp[i]["evidence"] = verdict, evidence

        # H1
        gaps = []
        for h in self.H:
            k = eff.get(h, {}).get("kev_zero", {}).get("start")
            ppo = [eff.get(h, {}).get(a, {}).get("start") for a in CLASSIC]
            ppo = [x for x in ppo if x is not None]
            if k is not None and ppo:
                gaps.append(k - max(ppo))
        if gaps:
            g = float(np.mean(gaps))
            put(
                "H1",
                "supported" if g >= 20 else "not supported" if g <= 0 else "inconclusive",
                f"Kev zero-shot starts {g:+.1f} points of % optimal vs the best untrained PPO (mean over horizons).",
            )
        # H2
        slower, total, crossed = 0, 0, False
        for h in self.H:
            e = eff.get(h, {})
            k = e.get("kev_rl", {}).get("gain_per_decade")
            ppo = [e.get(a, {}).get("gain_per_decade") for a in CLASSIC if a in e]
            ppo = [x for x in ppo if x is not None and np.isfinite(x)]
            if k is None or not np.isfinite(k) or not ppo:
                continue
            total += 1
            slower += int(k < max(ppo))
            crossed |= any(v is not None for key, v in e.get("_crossover", {}).items() if key.endswith(">kev_rl"))
        if total:
            v = "supported" if slower > total / 2 and crossed else "not supported" if slower == 0 else "inconclusive"
            put(
                "H2",
                v,
                f"Kev+RL learned more slowly than the best PPO in {slower} of {total} horizons; "
                f"PPO {'did' if crossed else 'did not'} cross Kev+RL's curve.",
            )
        # H3
        kev_arms = [a for a in ("kev_zero", "kev_icl") if all(pct(h, a) is not None for h in self.H)]
        if kev_arms and len(self.H) > 1:
            gap = {h: 100 - float(np.mean([pct(h, a) for a in kev_arms])) for h in self.H}
            delta = gap[max(self.H)] - gap[min(self.H)]
            put(
                "H3",
                "supported" if delta >= 5 else "not supported" if delta <= 2 else "inconclusive",
                f"Kev's gap to optimal goes from {gap[min(self.H)]:.1f} (H={min(self.H)}) to "
                f"{gap[max(self.H)]:.1f} points (H={max(self.H)}).",
            )
        # H4
        best = None
        for a in SYSTEM_ONE:
            vals = [results[h][a] for h in self.H if a in results.get(h, {})]
            if vals and (best is None or np.mean([v["pct_optimal"]["mean"] for v in vals]) > best[1]):
                best = (a, float(np.mean([v["pct_optimal"]["mean"] for v in vals])), vals)
        if best:
            key = f"{self.cfg.eval.error_budget:.2f}"
            dec = float(np.mean([v["handoff"]["decidable"]["automation"][key] for v in best[2]]))
            jud = float(np.mean([v["handoff"]["judgement"]["automation"][key] for v in best[2]]))
            ece = float(np.mean([v["calibration"]["ece"] for v in best[2]]))
            diff = 100 * (dec - jud)
            v = "supported" if diff >= 10 and ece <= 0.10 else "not supported" if diff <= 0 else "inconclusive"
            put(
                "H4",
                v,
                f"{narrative.ARMS[best[0]]['label']} automates {100 * dec:.0f}% of decidable vs "
                f"{100 * jud:.0f}% of judgement calls at a {100 * self.cfg.eval.error_budget:.0f}% error budget (ECE {ece:.2f}).",
            )
        # H5
        if all(pct(h, a) is not None for h in self.H for a in ("kev_zero", "kev_icl", "kev_rl")):
            z = np.mean([pct(h, "kev_zero") for h in self.H])
            icl = float(np.mean([pct(h, "kev_icl") for h in self.H]) - z)
            rl = float(np.mean([pct(h, "kev_rl") for h in self.H]) - z)
            share = icl / rl if rl > 0 else float("inf") if icl > 0 else 0.0
            v = "supported" if icl > 0 and share >= 0.25 else "not supported" if icl <= 0 else "inconclusive"
            put("H5", v, f"Context adds {icl:+.1f} points over zero-shot; RL adds {rl:+.1f}.")
        for item in hyp.values():
            item.setdefault("verdict", "pending")
            item.setdefault("evidence", "Not enough runs yet.")
        return list(hyp.values())

    # ------------------------------------------------------------ build
    def build(self) -> dict[str, Any]:
        refs = {h: self.s.refs(h) for h in self.H}
        results = {h: self.results(h, refs[h]) for h in self.H}
        curves, eff = {}, {}
        for h in self.H:
            curves[h], eff[h] = self.learning(h)
        dial = {
            arm: [
                {
                    "horizon": h,
                    "pct_optimal": results[h][arm]["pct_optimal"]["mean"],
                    "ci": results[h][arm]["pct_optimal"]["ci"],
                    "inbox_value": results[h][arm]["inbox_value"]["mean"],
                }
                for h in self.H
                if arm in results[h]
            ]
            for arm in narrative.ARMS
        }
        w = self.cfg.world
        report = {
            "meta": {
                "title": "Fast and Slow",
                "subtitle": "Can a System One decision model take over a planner's decidable calls?",
                "profile": self.cfg.profile,
                "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "horizons": self.H,
                "seeds": list(self.cfg.seeds),
                "kev_run": "fake (smoke test)" if self.cfg.kev.fake else self.cfg.kev.run,
                "note_writer": self.s.bank.writer,
                "scenarios_per_horizon": self.cfg.eval.scenarios_per_horizon,
                "error_budget": self.cfg.eval.error_budget,
                "thresholds": list(self.cfg.eval.efficiency_thresholds),
                "world": w.model_dump(),
            },
            "arms": {a: d for a, d in narrative.ARMS.items() if any(a in results[h] for h in self.H)},
            "references": refs,
            "results": results,
            "curves": curves,
            "efficiency": eff,
            "horizon_dial": {a: v for a, v in dial.items() if v},
            "handoff_budgets": [0.01, 0.02, self.cfg.eval.error_budget, 0.1],
            "costs": self.costs(),
            "replays": self.replays(),
            "desk": self.desk_example(),
            "hypotheses": self.verdicts(results, eff),
            "features": narrative.FEATURES,
            "decisions": narrative.DECISIONS,
            "limitations": narrative.LIMITATIONS,
        }
        return _clean(report)


def export(study: Study, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ReportBuilder(study).build(), indent=1) + "\n")
    return out
