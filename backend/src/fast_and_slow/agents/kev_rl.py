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

The trainer is policy-agnostic: the same code trains Kev on a GPU and the fake policy in the tests.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import KevConfig, WorldConfig
from ..notes.bank import NoteBank
from ..progress import Progress, fmt_duration
from ..progress import log as log_
from ..sim.desk import Episode
from ..sim.render import ORDER_Q, request
from ..sim.world import sample_scenario
from .policy import DecisionPolicy


def _write_synced(path: Path, text: str) -> None:
    """Write and fsync, so the file is complete on disk before anything that depends on it is written."""
    with path.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _fsync(path: Path) -> None:
    with path.open("rb+") as f:
        os.fsync(f.fileno())


def _problem(folder: Path) -> str | None:
    """Why this checkpoint folder can't be resumed from, or None when it can.

    trainer.json is written last and lists every other file with its size; all must be there at that size.
    Checkpoints from before the list existed are checked for their files being present and non-empty."""
    try:
        state = json.loads((folder / "trainer.json").read_text())
    except OSError:
        return "trainer.json is missing"
    except ValueError:
        return "trainer.json is not valid JSON"
    files = state.get("files") if isinstance(state, dict) else None
    if files is None:
        weights = [f for f in (folder / "policy").glob("*") if f.is_file()]
        if not weights:
            return "policy/ has no weights file"
        files = {f.relative_to(folder).as_posix(): None for f in [*weights, folder / "optimizer.pt"]}
    for name, size in files.items():
        f = folder / name
        if not f.exists():
            return f"{name} is missing"
        got = f.stat().st_size
        if (size is None and got == 0) or (size is not None and got != size):
            return f"{name} has {got:,} of {size if size is not None else 'some'} bytes"
    return None


def _is_complete(folder: Path) -> bool:
    return _problem(folder) is None


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
    paused: bool = False  # stopped at a checkpoint because the session's time budget ran out


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
                self.opt.zero_grad()
                # Gradient accumulation: each micro-batch's loss is its share of the minibatch mean, so the summed
                # gradient equals one pass over the whole minibatch while only rl_micro_batch requests sit in memory.
                loss_sum = ent_sum = clip_sum = 0.0
                step = max(1, cfg.rl_micro_batch)
                for k in range(0, len(mb), step):
                    part = mb[k : k + step]
                    logits = self.policy.score([s.req for s in part], grad=True)
                    logp_all = torch.stack([torch.log_softmax(z[ORDER_Q].float(), -1) for z in logits])
                    act = torch.tensor([s.action for s in part], device=logp_all.device)
                    logp = logp_all.gather(1, act[:, None]).squeeze(1)
                    old = torch.tensor([s.old_logp for s in part], dtype=logp.dtype, device=logp.device)
                    adv = torch.tensor([s.advantage for s in part], dtype=logp.dtype, device=logp.device)
                    ratio = torch.exp(logp - old)
                    clipped = torch.clamp(ratio, 1 - cfg.rl_clip, 1 + cfg.rl_clip)
                    pg = -torch.min(ratio * adv, clipped * adv).sum()
                    entropy = -(logp_all.exp() * logp_all).sum()
                    loss = (pg - cfg.rl_entropy * entropy) / len(mb)
                    loss.backward()  # type: ignore[no-untyped-call]
                    loss_sum += float(loss.detach())
                    ent_sum += float(entropy.detach())
                    clip_sum += float(((ratio.detach() - 1).abs() > cfg.rl_clip).float().sum())
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                self.opt.step()
                losses.append(loss_sum)
                ents.append(ent_sum / len(mb))
                clips.append(clip_sum / len(mb))
        return {"loss": float(np.mean(losses)), "entropy": float(np.mean(ents)), "clip_frac": float(np.mean(clips))}

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: Path, update: int, log: RLLog) -> None:
        """Everything needed to continue exactly where we stopped.

        Built for Google Drive's mount, where renaming a folder is unreliable and each file uploads on its own, after
        the write returns (a small file can reach Drive while a large one is still uploading, or never does if the
        runtime is deleted first). So:
        - each save goes into a new folder (`u0012/`) and its `trainer.json`, written last, lists every other file in
          the folder with its size: a folder counts only if all of them are there at full size;
        - `latest.json` then points at it;
        - the previous checkpoint is kept, and older ones deleted, so if the newest never fully reaches Drive the
          next session resumes from the one before (which had a whole save interval to upload)."""
        path.mkdir(parents=True, exist_ok=True)
        # A name is never reused (update number + save time): on Drive, deleting a folder and creating one with the
        # same name can leave two folders of that name, and the mount may show the stale one.
        folder = path / f"u{update:04d}-{time.time_ns() // 1000:016d}"
        folder.mkdir()
        self.policy.save(folder / "policy")
        torch.save({"opt": self.opt.state_dict()}, folder / "optimizer.pt")
        files = {}
        for f in sorted(x for x in folder.rglob("*") if x.is_file()):
            _fsync(f)
            files[f.relative_to(folder).as_posix()] = f.stat().st_size
        state = {
            "update": update,
            "decisions": self.decisions,
            "seconds": self.seconds,
            "rng": self.rng.bit_generator.state,
            "log": {"points": log.points, "updates": log.updates},
            "files": files,
        }
        _write_synced(folder / "trainer.json", json.dumps(state))
        _write_synced(path / "latest.json", json.dumps({"folder": folder.name, "update": update}))
        log_.info("checkpoint saved at update %s (%.0f MB to upload to Drive)", update, sum(files.values()) / 2**20)

        previous = [f for f in self._candidates(path) if f not in (folder, path) and _is_complete(f)][:1]
        for old in path.glob("u*"):
            if old.is_dir() and old != folder and old not in previous:
                shutil.rmtree(old, ignore_errors=True)
        if previous:  # the pre-folder layout goes only once two newer checkpoints exist
            for legacy in ("policy", "optimizer.pt", "trainer.json"):
                target = path / legacy
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink()

    @staticmethod
    def _candidates(path: Path) -> list[Path]:
        """Checkpoint folders, newest first: latest.json's, then every u*/ folder (`u0012-<time>`, or `u0012` from
        before names carried the time; both sort by update, then time), then the pre-folder layout."""
        out: list[Path] = []
        pointer = path / "latest.json"
        if pointer.exists():
            with contextlib.suppress(ValueError, KeyError, TypeError):
                out.append(path / json.loads(pointer.read_text())["folder"])
        out += sorted((f for f in path.glob("u*") if f.is_dir()), reverse=True)
        if (path / "trainer.json").exists():
            out.append(path)
        return list(dict.fromkeys(out))

    @staticmethod
    def _complete_checkpoint(path: Path) -> Path | None:
        """The newest checkpoint folder whose files all arrived (see save_checkpoint), or None."""
        return next((f for f in KevRLTrainer._candidates(path) if _is_complete(f)), None)

    def load_checkpoint(self, path: Path) -> tuple[int, RLLog] | None:
        candidates = self._candidates(path)
        complete = [f for f in candidates if _is_complete(f)]
        for bad in candidates[: candidates.index(complete[0])] if complete else candidates:
            log_.warning("checkpoint %s is incomplete, skipping it: %s", bad.name, _problem(bad))
        errors = []
        for folder in complete:
            try:
                state = json.loads((folder / "trainer.json").read_text())
                self.policy.load(folder / "policy")
                self.opt.load_state_dict(torch.load(folder / "optimizer.pt", weights_only=False)["opt"])
            except Exception as e:  # a file that has the right size but can't be read: try the one before
                log_.warning("checkpoint %s could not be read (%s); trying an older one", folder.name, e)
                errors.append(e)
                continue
            self.rng.bit_generator.state = state["rng"]
            self.decisions, self.seconds = int(state["decisions"]), float(state["seconds"])
            return int(state["update"]), RLLog(state["log"]["points"], state["log"]["updates"])
        if errors:  # weights may be half-loaded: training on from here would be neither fresh nor resumed
            raise RuntimeError(f"no checkpoint in {path} could be read; delete the folder to start over") from errors[
                -1
            ]
        if candidates:
            log_.warning("no complete checkpoint in %s: starting this run from update 0", path)
        return None

    # ------------------------------------------------------------ loop
    def train(
        self,
        evaluate: EvalFn,
        on_point: Callable[[RLLog], None] | None = None,
        checkpoint: Path | None = None,
        deadline: float | None = None,
        label: str = "kev_rl",
    ) -> RLLog:
        """deadline: a time.monotonic() value. When an update finishes past it, save a checkpoint and stop (paused)."""
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
        prog = Progress(label, self.cfg.rl_updates, "updates", every=0, done=start)
        if deadline is not None:
            log_.info("%s: will pause at a checkpoint after %s", label, fmt_duration(deadline - time.monotonic()))
        for u in range(start + 1, self.cfg.rl_updates + 1):
            t0 = time.perf_counter()
            batch = self.rollout()
            stats = self.update(batch)
            self.seconds += time.perf_counter() - t0
            mean_ret = float(np.mean([s.reward for s in batch])) / self.cfg.rl_reward_scale * self.H
            log.updates.append({"update": u, "decisions": self.decisions, "mean_return": mean_ret, **stats})
            note = f"mean return {'-' if mean_ret < 0 else ''}${abs(mean_ret):,.0f}"
            if u % self.cfg.eval_every_updates == 0 or u == self.cfg.rl_updates:
                log.points.append(
                    {"decisions": self.decisions, "score": evaluate(self.policy), "seconds": self.seconds}
                )
                note += f", held-out score {log.points[-1]['score']:.1f}% of optimal"
                if on_point:
                    on_point(log)
            if deadline is not None and u < self.cfg.rl_updates:
                note += f", pauses in {fmt_duration(deadline - time.monotonic())}"
            prog.update(u, note)
            out_of_time = deadline is not None and time.monotonic() >= deadline and u < self.cfg.rl_updates
            if checkpoint and (u % self.cfg.checkpoint_every_updates == 0 or u == self.cfg.rl_updates or out_of_time):
                self.save_checkpoint(checkpoint, u, log)
            if out_of_time:
                log.paused = True
                break
        return log
