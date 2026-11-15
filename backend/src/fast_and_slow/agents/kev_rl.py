"""RL fine-tuning of a decision model's choice head: PPO with group-relative advantages.

Kev never generates text. For the order question its pointer head returns a distribution over the order menu, and that
distribution *is* the policy. So the RL loop is PPO on those logits:

1. Rollout (no gradients): pick `groups_per_update` training scenario seeds; run `group_size` episodes on each seed in
   lockstep, sampling orders from the policy. Episodes on one seed share every exogenous draw (regimes, demand, notes),
   so differences in return come only from the decisions.
2. Advantage: return-to-go of each step minus the mean return-to-go of the same week across the group (GRPO-style
   baseline, no critic), normalised over the batch.
3. Update: `epochs` passes of clipped-ratio PPO over minibatches, recomputing log-probs with gradients, plus an entropy
   bonus. Adapter and head get separate learning rates.

The trainer is policy-agnostic: the same code trains Kev on an H100 and the fake policy in the tests.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import KevConfig, WorldConfig
from ..notes.bank import NoteBank
from ..sim.desk import Episode
from ..sim.render import ORDER_Q, request
from ..sim.world import sample_scenario
from .policy import DecisionPolicy

EvalFn = Callable[[DecisionPolicy], float]  # returns % of optimal on the held-out set


@dataclass
class Step:
    req: dict[str, object]
    action: int
    old_logp: float
    reward: float
    group: int
    week: int
    advantage: float = 0.0


@dataclass
class RLLog:
    points: list[dict[str, float]] = field(default_factory=list)  # decisions, score, seconds
    updates: list[dict[str, float]] = field(default_factory=list)  # loss, entropy, mean return, clip fraction


class KevRLTrainer:
    def __init__(
        self,
        world: WorldConfig,
        bank: NoteBank,
        horizon: int,
        policy: DecisionPolicy,
        cfg: KevConfig,
        seed: int = 0,
        train_seed_range: int = 10_000,
    ):
        self.w, self.bank, self.H, self.policy, self.cfg = world, bank, horizon, policy, cfg
        self.rng = np.random.default_rng(seed)
        self.train_seed_range = train_seed_range
        torch.manual_seed(seed)
        groups = []
        for g in policy.param_groups():
            lr = cfg.rl_head_lr if g["role"] == "head" else cfg.rl_lr
            groups.append({"params": g["params"], "lr": lr})
        self.opt = torch.optim.AdamW(groups, weight_decay=0.0)
        self.decisions = 0
        self.seconds = 0.0

    # ------------------------------------------------------------ rollout
    def rollout(self) -> list[Step]:
        seeds = self.rng.integers(0, self.train_seed_range, size=self.cfg.rl_groups_per_update)
        eps: list[Episode] = []
        group: list[int] = []
        for gi, s in enumerate(seeds):
            sc = sample_scenario(self.w, self.H, int(s))
            texts = self.bank.texts(sc, "train")
            for _ in range(self.cfg.rl_group_size):
                eps.append(Episode(self.w, sc, texts))
                group.append(gi)
        steps: list[list[Step]] = [[] for _ in eps]
        while not all(e.done for e in eps):
            live = [i for i, e in enumerate(eps) if not e.done]
            reqs = [self._order_request(eps[i]) for i in live]
            logits = self.policy.score(reqs)
            for i, req, z in zip(live, reqs, logits, strict=True):
                logp = torch.log_softmax(z[ORDER_Q].float(), -1)
                p = logp.exp().double().cpu().numpy()
                a = int(self.rng.choice(len(p), p=p / p.sum()))  # numpy RNG only: its state is checkpointed
                week = eps[i].week
                out = eps[i].step(self.w.order_menu[a])
                steps[i].append(Step(req, a, float(logp[a]), out.reward * self.cfg.rl_reward_scale, group[i], week))
        self.decisions += sum(len(s) for s in steps)
        return self._advantages(steps, group)

    def _order_request(self, ep: Episode) -> dict[str, object]:
        r = request(self.w, ep.observe())
        r["questions"] = {ORDER_Q: r["questions"][ORDER_Q]}
        return r

    def _advantages(self, steps: list[list[Step]], group: list[int]) -> list[Step]:
        rtg: dict[tuple[int, int], list[float]] = {}
        for ep_steps in steps:
            g = 0.0
            for st in reversed(ep_steps):
                g += st.reward
                st.advantage = g  # return-to-go for now
                rtg.setdefault((st.group, st.week), []).append(g)
        flat = [st for ep_steps in steps for st in ep_steps]
        for st in flat:
            st.advantage -= float(np.mean(rtg[(st.group, st.week)]))
        adv = np.asarray([st.advantage for st in flat])
        std = adv.std() + 1e-8
        for st in flat:
            st.advantage /= std
        return flat

    # ------------------------------------------------------------ update
    def update(self, batch: list[Step]) -> dict[str, float]:
        cfg = self.cfg
        params = [p for g in self.opt.param_groups for p in g["params"]]
        losses, ents, clips = [], [], []
        for _ in range(cfg.rl_epochs):
            order = self.rng.permutation(len(batch))
            for i in range(0, len(order), cfg.rl_minibatch):
                mb = [batch[j] for j in order[i : i + cfg.rl_minibatch]]
                logits = self.policy.score([s.req for s in mb], grad=True)
                logp_all = torch.stack([torch.log_softmax(z[ORDER_Q].float(), -1) for z in logits])
                act = torch.tensor([s.action for s in mb])
                logp = logp_all.gather(1, act[:, None]).squeeze(1)
                old = torch.tensor([s.old_logp for s in mb], dtype=logp.dtype, device=logp.device)
                adv = torch.tensor([s.advantage for s in mb], dtype=logp.dtype, device=logp.device)
                ratio = torch.exp(logp - old)
                clipped = torch.clamp(ratio, 1 - cfg.rl_clip, 1 + cfg.rl_clip)
                pg = -torch.min(ratio * adv, clipped * adv).mean()
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pg - cfg.rl_entropy * entropy
                self.opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                self.opt.step()
                losses.append(float(loss.detach()))
                ents.append(float(entropy.detach()))
                clips.append(float(((ratio.detach() - 1).abs() > cfg.rl_clip).float().mean()))
        return {"loss": float(np.mean(losses)), "entropy": float(np.mean(ents)), "clip_frac": float(np.mean(clips))}

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: Path, update: int, log: RLLog) -> None:
        """Everything needed to continue exactly where we stopped. Written to a temp dir, then swapped in, so a
        disconnect mid-write never leaves a half checkpoint behind."""
        tmp, old = path.with_name(path.name + ".tmp"), path.with_name(path.name + ".old")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.policy.save(tmp / "policy")
        torch.save({"opt": self.opt.state_dict()}, tmp / "optimizer.pt")
        state = {
            "update": update,
            "decisions": self.decisions,
            "seconds": self.seconds,
            "rng": self.rng.bit_generator.state,
            "log": {"points": log.points, "updates": log.updates},
        }
        (tmp / "trainer.json").write_text(json.dumps(state))
        if path.exists():
            shutil.rmtree(old, ignore_errors=True)
            path.rename(old)
        tmp.rename(path)
        shutil.rmtree(old, ignore_errors=True)

    def load_checkpoint(self, path: Path) -> tuple[int, RLLog] | None:
        if not (path / "trainer.json").exists():
            return None
        state = json.loads((path / "trainer.json").read_text())
        self.policy.load(path / "policy")
        self.opt.load_state_dict(torch.load(path / "optimizer.pt", weights_only=False)["opt"])
        self.rng.bit_generator.state = state["rng"]
        self.decisions, self.seconds = int(state["decisions"]), float(state["seconds"])
        return int(state["update"]), RLLog(state["log"]["points"], state["log"]["updates"])

    # ------------------------------------------------------------ loop
    def train(
        self,
        evaluate: EvalFn,
        on_point: Callable[[RLLog], None] | None = None,
        checkpoint: Path | None = None,
    ) -> RLLog:
        resumed = self.load_checkpoint(checkpoint) if checkpoint else None
        if resumed:
            start, log = resumed
            log_msg = f"resumed from update {start} ({self.decisions} decisions)"
            log.updates.append({"update": start, "event": log_msg})  # type: ignore[dict-item]
        else:
            start, log = 0, RLLog()
            log.points.append({"decisions": 0, "score": evaluate(self.policy), "seconds": 0.0})
            if on_point:
                on_point(log)
        for u in range(start + 1, self.cfg.rl_updates + 1):
            t0 = time.perf_counter()
            batch = self.rollout()
            stats = self.update(batch)
            self.seconds += time.perf_counter() - t0
            mean_ret = float(np.mean([s.reward for s in batch])) / self.cfg.rl_reward_scale * self.H
            log.updates.append({"update": u, "decisions": self.decisions, "mean_return": mean_ret, **stats})
            if u % self.cfg.eval_every_updates == 0 or u == self.cfg.rl_updates:
                log.points.append(
                    {"decisions": self.decisions, "score": evaluate(self.policy), "seconds": self.seconds}
                )
                if on_point:
                    on_point(log)
            if checkpoint and (u % self.cfg.checkpoint_every_updates == 0 or u == self.cfg.rl_updates):
                self.save_checkpoint(checkpoint, u, log)
        return log
