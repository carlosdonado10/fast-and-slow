"""Classic RL arms: PPO (stable-baselines3, MLP policy) on the desk, with three observation variants (agents/features.py).

Training scenarios are drawn from seeds below the held-out range and use the *train* note wording; evaluation uses the
held-out scenarios and the *test* wording, exactly as for Kev. Every environment step is one planning decision, so
SB3's timestep counter is the x axis of the learning curve.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ..config import WorldConfig
from ..notes.bank import NoteBank
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
    def __init__(self, checkpoints: list[int], on_checkpoint: Callable[[int], None]):
        super().__init__()
        self.pending = list(checkpoints)
        self.on_checkpoint = on_checkpoint

    def _on_step(self) -> bool:
        while self.pending and self.num_timesteps >= self.pending[0]:
            self.pending.pop(0)
            self.on_checkpoint(self.num_timesteps)
        return True


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
) -> tuple[PPO, list[dict[str, float]]]:
    """Train PPO and return the model and its learning curve [{decisions, score, seconds}]."""

    def make(i: int) -> Callable[[], DeskEnv]:
        return lambda: DeskEnv(world, bank, horizon, variant, embedder, seed=seed * 1000 + i)

    env = DummyVecEnv([make(i) for i in range(n_envs)])
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
    curve: list[dict[str, float]] = []
    started = time.perf_counter()
    train_time = [0.0]

    def checkpoint(decisions: int) -> None:
        train_time[0] = time.perf_counter() - started - sum(p.get("eval_seconds", 0.0) for p in curve)
        t0 = time.perf_counter()
        score = evaluate(PPOAgent(name, world, model, variant, embedder))
        curve.append(
            {"decisions": decisions, "score": score, "seconds": train_time[0], "eval_seconds": time.perf_counter() - t0}
        )

    checkpoint(0)
    model.learn(
        total_timesteps=total_decisions,
        callback=_CurveCallback(log_checkpoints(total_decisions, eval_points), checkpoint),
    )
    if not curve or curve[-1]["decisions"] < model.num_timesteps:
        checkpoint(int(model.num_timesteps))
    return model, curve
