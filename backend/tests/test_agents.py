from __future__ import annotations

import numpy as np
import pytest
import torch

from fast_and_slow.agents.classic_rl import DeskEnv, PPOAgent, log_checkpoints, train_ppo
from fast_and_slow.agents.features import CachedEmbedder, HashEmbedder, feature_dim, features
from fast_and_slow.agents.kev_agent import KevAgent, icl_context
from fast_and_slow.agents.kev_rl import KevRLTrainer
from fast_and_slow.agents.policy import FakePolicy, hashed_bag, load_policy
from fast_and_slow.config import StudyConfig
from fast_and_slow.eval.runner import Labeller, eval_scenarios, evaluate
from fast_and_slow.notes.bank import NoteBank
from fast_and_slow.sim.desk import Episode
from fast_and_slow.sim.render import ORDER_Q, REGIME_Q, request
from fast_and_slow.sim.world import sample_scenario


def obs_for(cfg: StudyConfig, bank: NoteBank, weeks: int = 2):  # type: ignore[no-untyped-def]
    sc = sample_scenario(cfg.world, 4, 11)
    ep = Episode(cfg.world, sc, bank.texts(sc, "train"))
    for _ in range(weeks):
        ep.step(10)
    return ep.observe()


def test_request_has_the_menu_in_order_and_never_leaks_the_regime(cfg: StudyConfig, bank: NoteBank) -> None:
    o = obs_for(cfg, bank)
    r = request(cfg.world, o, context="EXAMPLES")
    keys = list(r["questions"][ORDER_Q]["criteria"])
    assert keys == [f"order_{q}" for q in cfg.world.order_menu]
    assert set(r["questions"][REGIME_Q]["criteria"]) == {"low", "normal", "surge"}
    assert "EXAMPLES" in r["state"] and "Week 3 of 4" in r["state"]
    for word in ("regime", "surge regime", "true"):
        assert word not in r["state"].lower().split()


def test_fake_policy_scores_every_option_and_is_trainable(cfg: StudyConfig, bank: NoteBank) -> None:
    pol = FakePolicy(seed=1)
    req = request(cfg.world, obs_for(cfg, bank))
    out = pol.score([req])[0]
    assert out[ORDER_Q].shape == (len(cfg.world.order_menu),) and out[REGIME_Q].shape == (3,)
    assert not out[ORDER_Q].requires_grad
    assert pol.score([req], grad=True)[0][ORDER_Q].requires_grad
    assert {g["role"] for g in pol.param_groups()} == {"adapter", "head"}
    assert hashed_bag("", 8).sum() == 0
    assert isinstance(load_policy("x", fake=True), FakePolicy)


def test_kev_agent_returns_distributions(cfg: StudyConfig, bank: NoteBank) -> None:
    agent = KevAgent("kev_zero", cfg.world, FakePolicy(), batch_size=2)
    o = obs_for(cfg, bank)
    ds = agent.act_batch([o, o, o], [0, 1, 2])
    assert len(ds) == 3
    for d in ds:
        assert np.isclose(d.probs.sum(), 1) and d.regime_probs is not None and 0 <= d.confidence <= 1
    no_regime = KevAgent("k", cfg.world, FakePolicy(), ask_regime=False).act_batch([o], [0])[0]
    assert no_regime.regime_probs is None


def test_icl_context_lists_k_varied_examples(cfg: StudyConfig, bank: NoteBank) -> None:
    ctx = icl_context(cfg.world, bank, 4, k=4)
    lines = [ln for ln in ctx.splitlines() if ln.startswith("- Week")]
    assert len(lines) == 4
    assert "ordered" in lines[0] and "profit that week" in lines[0]


def test_rl_trainer_advantages_are_group_relative_and_update_moves_weights(cfg: StudyConfig, bank: NoteBank) -> None:
    pol = FakePolicy(seed=3)
    tr = KevRLTrainer(cfg.world, bank, 3, pol, cfg.kev, seed=0)
    batch = tr.rollout()
    assert len(batch) == cfg.kev.rl_groups_per_update * cfg.kev.rl_group_size * 3
    assert tr.decisions == len(batch)
    for g in range(cfg.kev.rl_groups_per_update):
        adv = [s.advantage for s in batch if s.group == g and s.week == 0]
        assert abs(np.mean(adv)) < 1e-6
    before = pol.state_proj.detach().clone()
    stats = tr.update(batch)
    assert not torch.equal(before, pol.state_proj.detach())
    assert set(stats) == {"loss", "entropy", "clip_frac"}


def test_rl_training_loop_logs_a_curve(cfg: StudyConfig, bank: NoteBank) -> None:
    calls: list[int] = []
    tr = KevRLTrainer(cfg.world, bank, 2, FakePolicy(), cfg.kev, seed=1)
    log = tr.train(lambda p: float(len(calls)), on_point=lambda lg: calls.append(len(lg.points)))
    assert log.points[0]["decisions"] == 0
    assert log.points[-1]["decisions"] == tr.decisions > 0
    assert len(log.updates) == cfg.kev.rl_updates


def test_feature_variants_have_the_declared_sizes(cfg: StudyConfig, bank: NoteBank) -> None:
    o = obs_for(cfg, bank, weeks=3)
    emb = CachedEmbedder(HashEmbedder(16))
    for v in ("num", "sig", "emb"):
        x = features(cfg.world, o, v, emb)
        assert x.shape == (feature_dim(cfg.world, v, 16),) and np.isfinite(x).all()
    with pytest.raises(ValueError):
        features(cfg.world, o, "emb")
    with pytest.raises(ValueError):
        features(cfg.world, o, "pixels")
    assert emb.embed([]).shape == (0, 16)
    v1, v2 = emb.embed(["same text", "same text"])
    assert np.allclose(v1, v2) and np.isclose(np.linalg.norm(v1), 1)


def test_desk_env_runs_an_episode(cfg: StudyConfig, bank: NoteBank) -> None:
    env = DeskEnv(cfg.world, bank, 3, "sig", None, seed=0)
    obs, _ = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    total, done = 0.0, False
    while not done:
        obs, r, done, _, _ = env.step(2)
        total += r
    assert np.isfinite(total)


def test_log_checkpoints_are_increasing_and_end_at_total() -> None:
    pts = log_checkpoints(10_000, 5)
    assert pts == sorted(pts) and pts[-1] == 10_000 and pts[0] == 100


def test_train_ppo_returns_a_curve_and_an_agent(cfg: StudyConfig, bank: NoteBank) -> None:
    w, h = cfg.world, 2
    lab = Labeller(w, h, cfg.oracle)
    sc = eval_scenarios(w, h, 4, 10_000)
    model, curve = train_ppo(w, bank, h, "num", None, 1_000, 2, 3, lambda a: 0.0, seed=0)
    assert curve[0]["decisions"] == 0 and curve[-1]["decisions"] >= 1_000
    r = evaluate(PPOAgent("ppo_num", w, model, "num", None), w, bank, sc, lab)
    assert len(r.decisions) == 4 * h
