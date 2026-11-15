"""Exact-ish Bayes-optimal planning by dynamic programming over (period, inventory, belief).

The belief over regimes lives on a simplex grid (resolution R: (R+1)(R+2)/2 points; R=20 gives 231). Values are solved
backwards on the grid; at run time the policy evaluates the one-step lookahead from the *exact* belief, reading future
values from the grid. That keeps decisions sharp while the table stays small.

Timeline of a period t:  belief b⁻ → note n_t (update) → order q → demand d_t (update) → predict with T → b⁻ of t+1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..config import OracleConfig, WorldConfig
from ..sim.world import Note, demand_pmf, note_likelihood, note_outcomes, transition_matrix


def simplex_grid(k: int, resolution: int) -> np.ndarray:
    if k != 3:
        raise NotImplementedError("grid is written for 3 regimes")
    pts = [(i, j, resolution - i - j) for i in range(resolution + 1) for j in range(resolution + 1 - i)]
    return np.asarray(pts, dtype=float) / resolution


def _normalize(b: np.ndarray) -> np.ndarray:
    s = b.sum(axis=-1, keepdims=True)
    return np.where(s > 0, b / np.where(s > 0, s, 1), 1.0 / b.shape[-1])


@dataclass
class Tables:
    grid: np.ndarray  # [G, K]
    index: dict[tuple[int, int, int], int]
    resolution: int

    def nearest(self, b: np.ndarray) -> np.ndarray:
        """Nearest grid point(s) by largest-remainder rounding. b: [..., K] -> int indices [...]"""
        b = _normalize(np.asarray(b, dtype=float))
        scaled = b * self.resolution
        base = np.floor(scaled).astype(int)
        short = self.resolution - base.sum(axis=-1)
        order = np.argsort(-(scaled - base), axis=-1)
        bump = np.zeros_like(base)
        ranks = np.argsort(order, axis=-1)
        bump[ranks < short[..., None]] = 1
        pts = base + bump
        flat = pts.reshape(-1, pts.shape[-1])
        return np.asarray([self.index[tuple(p)] for p in flat]).reshape(pts.shape[:-1])


class BayesPlanner:
    """Solves the finite-horizon POMDP for one horizon and exposes Q-values, the belief filter and decidability."""

    def __init__(self, world: WorldConfig, horizon: int, cfg: OracleConfig | None = None):
        self.w, self.H, self.cfg = world, horizon, cfg or OracleConfig()
        self.K = len(world.regimes)
        self.T = transition_matrix(world)
        self.pmf = demand_pmf(world)  # [K, D]
        self.D = world.max_demand + 1
        self.menu = np.asarray(world.order_menu)
        self.I = world.inventory_cap + 1
        grid = simplex_grid(self.K, self.cfg.belief_resolution)
        idx = {tuple(np.rint(p * self.cfg.belief_resolution).astype(int)): g for g, p in enumerate(grid)}
        self.tab = Tables(grid, idx, self.cfg.belief_resolution)
        self._economics()
        self._solve()

    # ---------- economics, independent of belief ----------
    def _economics(self) -> None:
        w = self.w
        inv = np.arange(self.I)[:, None, None]
        q = np.minimum(self.menu[None, :, None], w.inventory_cap - inv)
        d = np.arange(self.D)[None, None, :]
        on_hand = inv + q
        sales = np.minimum(on_hand, d)
        left = on_hand - sales
        base = w.price * sales - w.unit_cost * q - w.fixed_order_cost * (q > 0) - w.holding_cost * left
        base = base - w.stockout_penalty * (d - sales)
        self.r_mid = base.astype(float)  # [I, A, D]
        self.r_last = base + w.salvage_value * left
        self.next_inv = left  # [I, A, D]

    # ---------- belief tables on the grid ----------
    @cached_property
    def p_demand(self) -> np.ndarray:  # [G, D]
        return self.tab.grid @ self.pmf

    @cached_property
    def predict_idx(self) -> np.ndarray:  # [G, D] grid index of b⁻ next period after seeing demand d
        post = _normalize(self.tab.grid[:, None, :] * self.pmf.T[None, :, :])  # [G, D, K]
        return self.tab.nearest(post @ self.T)

    @cached_property
    def notes(self) -> list[tuple[Note, np.ndarray]]:
        return note_outcomes(self.w)

    def _note_tables(self) -> tuple[np.ndarray, np.ndarray]:
        probs, idx = [], []
        for _, lik in self.notes:
            probs.append(self.tab.grid @ lik)  # [G]
            idx.append(self.tab.nearest(_normalize(self.tab.grid * lik)))
        return np.stack(probs, 1), np.stack(idx, 1)  # [G, N]

    # ---------- backward induction ----------
    def _q_grid(self, t: int, W_next: np.ndarray | None) -> np.ndarray:
        """Q[g, I, A] for every grid belief at period t (belief after the note)."""
        last = t == self.H - 1
        r = self.r_last if last else self.r_mid  # [I, A, D]
        pd = self.p_demand  # [G, D]
        q = np.einsum("gd,iad->gia", pd, r)
        if W_next is not None:
            fut = W_next[self.next_inv[None, :, :, :], self.predict_idx[:, None, None, :]]  # [G, I, A, D]
            q = q + np.einsum("gd,giad->gia", pd, fut)
        return q

    def _solve(self) -> None:
        note_p, note_idx = self._note_tables()
        V_list: list[np.ndarray] = []  # V_t[I, g] after the note, filled backwards
        W_list: list[np.ndarray] = []  # W_t[I, g] before the note
        W_next: np.ndarray | None = None
        for t in reversed(range(self.H)):
            Q = self._q_grid(t, W_next)  # [G, I, A]
            V = Q.max(axis=2).T  # [I, G]
            V_list.append(V)
            # expectation over this period's note, from the pre-note belief
            W = np.einsum("gn,ign->ig", note_p, V[:, note_idx])
            W_list.append(W)
            W_next = W
        self.V = V_list[::-1]
        self.W = W_list[::-1]

    # ---------- run-time interface ----------
    def prior(self) -> np.ndarray:
        return np.asarray(self.w.regime_prior, dtype=float)

    def after_note(self, b_pre: np.ndarray, note: Note) -> np.ndarray:
        return _normalize(b_pre * note_likelihood(self.w, note))

    def after_demand(self, b: np.ndarray, demand: int) -> np.ndarray:
        """Belief update on the realized demand, then prediction to the next period."""
        return _normalize(b * self.pmf[:, demand]) @ self.T

    def q_values(self, t: int, inventory: int, b: np.ndarray) -> np.ndarray:
        """Q[a] for the exact belief b (after period t's note), one-step lookahead on the solved grid."""
        last = t == self.H - 1
        r = (self.r_last if last else self.r_mid)[inventory]  # [A, D]
        pd = b @ self.pmf  # [D]
        q = r @ pd
        if not last:
            post = _normalize(b[None, :] * self.pmf.T) @ self.T  # [D, K]
            nxt = self.tab.nearest(post)  # [D]
            fut = self.W[t + 1][self.next_inv[inventory], nxt[None, :]]  # [A, D]
            q = q + fut @ pd
        return q

    def value(self) -> float:
        """Expected total profit of the Bayes policy from the start (grid approximation)."""
        g0 = int(self.tab.nearest(self.prior()))
        return float(self.W[0][self.w.start_inventory, g0])

    def acceptable(self, q: np.ndarray) -> np.ndarray:
        """Boolean mask of actions within the regret tolerance of the best one."""
        return q >= q.max() - self.cfg.acceptable_regret_usd - 1e-9


class PerfectInfoPlanner:
    """Knows the current regime every period (not the future path, not demand). Upper bound on what reading can buy."""

    def __init__(self, world: WorldConfig, horizon: int):
        bp = BayesPlanner.__new__(BayesPlanner)
        bp.w, bp.H = world, horizon
        bp.menu, bp.D, bp.I = np.asarray(world.order_menu), world.max_demand + 1, world.inventory_cap + 1
        BayesPlanner._economics(bp)
        self.w, self.H = world, horizon
        self.T, self.pmf = transition_matrix(world), demand_pmf(world)
        self.r_mid, self.r_last, self.next_inv = bp.r_mid, bp.r_last, bp.next_inv
        K = len(world.regimes)
        Q_list: list[np.ndarray] = []  # Q_t[z, I, A], filled backwards
        V_next: np.ndarray | None = None  # V_{t+1}[I, z]
        for t in reversed(range(horizon)):
            r = self.r_last if t == horizon - 1 else self.r_mid
            Q = np.einsum("zd,iad->zia", self.pmf, r)
            if V_next is not None:
                cont = V_next @ self.T.T  # [I, z]: E over next regime given current z
                Q = Q + np.einsum("zd,iadz->zia", self.pmf, cont[self.next_inv][..., :K])
            Q_list.append(Q)
            V_next = Q.max(axis=2).T
        self.Q = Q_list[::-1]
        self.V0 = V_next

    def q_values(self, t: int, inventory: int, regime: int) -> np.ndarray:
        return self.Q[t][regime, inventory]

    def best_action(self, t: int, inventory: int, regime: int) -> int:
        return int(np.argmax(self.Q[t][regime, inventory]))


def is_decidable(
    bayes_action: int, b: np.ndarray, t: int, inventory: int, pi: PerfectInfoPlanner, floor: float
) -> bool:
    """Decidable: the Bayes action is also the full-information best action under every plausible regime."""
    plausible = np.flatnonzero(b >= floor)
    return all(pi.best_action(t, inventory, int(z)) == bayes_action for z in plausible)
