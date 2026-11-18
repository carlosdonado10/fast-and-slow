"""Classic RL arms: PPO (stable-baselines3, MLP policy) on the desk, with three observation variants (agents/features.py).

Training scenarios are drawn from seeds below the held-out range and use the *train* note wording; evaluation uses the
held-out scenarios and the *test* wording, exactly as for Kev. Every environment step is one planning decision, so
SB3's timestep counter is the x axis of the learning curve.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ..config import WorldConfig
from ..notes.bank import NoteBank
from ..progress import Progress
from ..sim.desk import Episode, Observation
from ..sim.world import sample_scenario
from .base import Agent, Decision
from .features import Embedder, feature_dim, features


class DeskEnv(gym.Env):
    metadata: dict[str, Any] = {}  # noqa: RUF012 - gymnasium declares it as a plain class attribute

    def __init__(
        self,
        world: WorldConfig,
        bank: NoteBank,
        horizon: int,
        variant: str,
        embedder: Embedder | None,
        seed: int,
        reward_scale: float = 0.01,
        seed_range: int = 10_000,
    ):
        super().__init__()
        self.w, self.bank, self.H, self.variant, self.embedder = world, bank, horizon, variant, embedder
        self.reward_scale, self.seed_range = reward_scale, seed_range
        dim = self.dim = feature_dim(world, variant, embedder.dim if embedder else 0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(world.order_menu))
        self.rng = np.random.default_rng(seed)
        self.ep: Episode | None = None

    def _obs(self) -> np.ndarray:
        assert self.ep is not None
        return features(self.w, self.ep.observe(), self.variant, self.embedder)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sc = sample_scenario(self.w, self.H, int(self.rng.integers(self.seed_range)))
        self.ep = Episode(self.w, sc, self.bank.texts(sc, "train"))
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.ep is not None
        out = self.ep.step(self.w.order_menu[int(action)])
        done = self.ep.done
        obs = np.zeros(self.dim, dtype=np.float32) if done else self._obs()
        return obs, out.reward * self.reward_scale, done, False, {}


class PPOAgent(Agent):
    def __init__(self, name: str, world: WorldConfig, model: PPO, variant: str, embedder: Embedder | None):
        self.name, self.w, self.model, self.variant, self.embedder = name, world, model, variant, embedder

    def act_batch(self, obs: list[Observation], live: list[int]) -> list[Decision]:
        t0 = time.perf_counter()
        x = np.stack([features(self.w, o, self.variant, self.embedder) for o in obs])
        with torch.no_grad():
            t, _ = self.model.policy.obs_to_tensor(x)
            dist = self.model.policy.get_distribution(t).distribution
            probs = dist.probs.cpu().numpy().astype(float)  # type: ignore[union-attr]
        per = (time.perf_counter() - t0) * 1000 / len(obs)
        return [Decision(int(np.argmax(p)), p, None, per) for p in probs]


def log_checkpoints(total: int, n: int, first: int = 100) -> list[int]:
    """n checkpoints spaced evenly in log10(decisions) from `first` to `total` (0 is always evaluated as well)."""
    pts = np.unique(np.round(np.logspace(np.log10(first), np.log10(total), n)).astype(int))
    return [int(p) for p in pts]


class _CurveCallback(BaseCallback):
    """Evaluates at the learning-curve checkpoints and saves a resumable checkpoint every `save_every` decisions."""

    def __init__(
        self,
        checkpoints: list[int],
        on_checkpoint: Callable[[int], None],
        save_every: int = 0,
        on_save: Callable[[], None] | None = None,
        progress: Progress | None = None,
    ):
        super().__init__()
        self.progress = progress
        self.pending = list(checkpoints)
        self.on_checkpoint, self.save_every, self.on_save = on_checkpoint, save_every, on_save
        self.last_save = 0

    def _on_step(self) -> bool:
        while self.pending and self.num_timesteps >= self.pending[0]:
            self.pending.pop(0)
            self.on_checkpoint(self.num_timesteps)
        if self.on_save and self.save_every and self.num_timesteps - self.last_save >= self.save_every:
            self.last_save = self.num_timesteps
            self.on_save()
        if self.progress:
            self.progress.update(self.num_timesteps)
        return True


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def train_ppo(
    world: WorldConfig,
    bank: NoteBank,
    horizon: int,
    variant: str,
    embedder: Embedder | None,
    total_decisions: int,
    n_envs: int,
    eval_points: int,
    evaluate: Callable[[Agent], float],
    seed: int = 0,
    checkpoint: Path | None = None,
    save_every: int = 25_000,
    init: Path | None = None,
    label: str = "",
) -> tuple[PPO, list[dict[str, float]]]:
    """Train PPO and return the model and its learning curve [{decisions, score, seconds}].

    With `checkpoint` (a directory), the model and the curve are saved every `save_every` decisions and at every curve
    point; a later call with the same directory continues from there. The policy, optimizer and step count resume
    exactly; which training scenarios come next does not (the env RNG restarts), which leaves the statistics unchanged.
    With `init` (a saved model.zip), a fresh run starts from that policy instead of from scratch (H6 retraining).
    """

    def make(i: int) -> Callable[[], DeskEnv]:
        return lambda: DeskEnv(world, bank, horizon, variant, embedder, seed=seed * 1000 + i)

    env = DummyVecEnv([make(i) for i in range(n_envs)])
    model_file = checkpoint / "model.zip" if checkpoint else None
    curve_file = checkpoint / "curve.json" if checkpoint else None
    curve: list[dict[str, float]] = []
    if model_file and curve_file and model_file.exists() and curve_file.exists():
        model = PPO.load(model_file, env=env, device="cpu")
        curve = json.loads(curve_file.read_text())
    elif init is not None:
        # warm start (H6 retraining): the old world's policy keeps learning in the new one, counting from zero
        model = PPO.load(init, env=env, device="cpu")
        model.num_timesteps = 0
        model._num_timesteps_at_start = 0
    else:
        n_steps = max(16, min(256, total_decisions // (n_envs * 4)))
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=min(256, n_steps * n_envs),
            learning_rate=3e-4,
            gamma=1.0,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [128, 128]},
            seed=seed,
            verbose=0,
            device="cpu",
        )
    name = f"ppo_{variant}"
    base_seconds = curve[-1]["seconds"] if curve else 0.0
    started = time.perf_counter()
    eval_time = [0.0]

    def train_seconds() -> float:
        return base_seconds + time.perf_counter() - started - eval_time[0]

    def save() -> None:
        if model_file and curve_file:
            model_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = model_file.with_name("model.tmp.zip")
            model.save(tmp)
            os.replace(tmp, model_file)
            _atomic_write(curve_file, json.dumps(curve))

    def point(decisions: int) -> None:
        t0 = time.perf_counter()
        score = evaluate(PPOAgent(name, world, model, variant, embedder))
        spent = time.perf_counter() - t0
        eval_time[0] += spent
        curve.append({"decisions": decisions, "score": score, "seconds": train_seconds(), "eval_seconds": spent})
        save()

    if not curve:
        point(0)
    done = int(model.num_timesteps)
    if done < total_decisions:
        pending = [c for c in log_checkpoints(total_decisions, eval_points) if c > done]
        prog = Progress(label or name, total_decisions, "decisions", done=done)
        model.learn(
            total_timesteps=total_decisions - done,
            reset_num_timesteps=False,
            callback=_CurveCallback(pending, point, save_every, save, prog),
        )
        prog.update(total_decisions, f"held-out score {curve[-1]['score']:.1f}% of optimal")
    if curve[-1]["decisions"] < model.num_timesteps:
        point(int(model.num_timesteps))
    return model, curve
