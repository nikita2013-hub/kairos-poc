"""Simulated learners (manuscript §5.1).

A simulated learner has a latent ability theta_v per skill (logit scale). Responses follow a
2PL model with guess/slip, attenuated by prerequisite deficits. Instruction moves ability
according to a learning curve that peaks when the item sits just above ability (proximal
challenge). Groups differ ONLY through injected generative differences (initial-ability
offsets, and optionally a latency baseline shift), so any group difference in the tutor's
*treatment* at matched state is attributable to the tutor, which is what the audit tests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .graph import SkillGraph
from .types import Move, MOVE_EFFICACY, Schema

SCHEMA_DIMS = {
    "referential_context": ["urban_market", "rural_farming", "cricket", "cooking", "public_transport"],
    "language_register": ["formal_english", "hinglish", "simple_english"],
    "framing": ["narrative_first", "definition_first"],
    "collaboration": ["individual", "collaborative"],
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SimConfig:
    n_groups: int = 2
    group_theta_offset: tuple[float, ...] = (0.0, -0.5)   # generative difference in prior ability
    group_latency_shift: tuple[float, ...] = (0.0, 0.3)   # log-latency baseline shift (behavioural confound)
    schema_group_corr: float = 0.8                        # P(schema value drawn from group-typical value)
    slope: float = 1.7
    guess: float = 0.15
    slip: float = 0.08
    prereq_penalty: float = 0.6
    zpd_peak: float = 0.5        # gain peaks when b - theta = zpd_peak
    zpd_width: float = 0.9
    adapted_difficulty_shift: float = 0.0   # >0 simulates presentation leaking into difficulty (DIF)


@dataclass
class SkillParams:
    b_center: np.ndarray     # typical item difficulty per skill (logit)
    theta0_mean: np.ndarray  # mean initial ability per skill
    theta0_sd: np.ndarray
    lr: np.ndarray           # learning rate per skill

    @staticmethod
    def default(n: int, graph: SkillGraph, seed: int = 0) -> "SkillParams":
        rng = np.random.default_rng(seed)
        depth = np.array([graph.depth(v) for v in range(n)])
        return SkillParams(b_center=rng.normal(0, 0.4, n) + 0.25 * depth,
                           theta0_mean=rng.normal(-0.2, 0.4, n) - 0.35 * depth,
                           theta0_sd=np.full(n, 0.8), lr=rng.uniform(0.15, 0.35, n))

    @staticmethod
    def calibrate_from_logs(df: pd.DataFrame, n: int) -> "SkillParams":
        """Moment-match per-skill first-attempt accuracy and the accuracy-by-attempt slope."""
        b_center = np.zeros(n)
        theta0, lr = np.zeros(n), np.full(n, 0.2)
        df = df.copy()
        df["attempt"] = df.groupby(["user", "skill"]).cumcount()
        for v in range(n):
            g = df[df.skill == v]
            if len(g) < 50:
                continue
            p0 = np.clip(g[g.attempt == 0].correct.mean(), 0.05, 0.95)
            theta0[v] = np.log(p0 / (1 - p0))
            by = g[g.attempt < 10].groupby("attempt").correct.mean().clip(0.05, 0.95)
            if len(by) >= 3:
                slope = np.polyfit(np.log1p(by.index.values), np.log(by / (1 - by)).values, 1)[0]
                lr[v] = float(np.clip(slope * 0.5, 0.05, 0.6))
        return SkillParams(b_center, theta0, np.full(n, 0.8), lr)


@dataclass
class SimLearner:
    learner_id: int
    group: int
    theta: np.ndarray
    schema: Schema
    rng: np.random.Generator
    cfg: SimConfig
    graph: SkillGraph
    lr: np.ndarray
    log_latency_base: float = 9.5
    frustration: float = 0.0

    def effective_theta(self, v: int) -> float:
        pre = self.graph.prereqs(v)
        deficit = np.mean([max(0.0, 0.5 - self.theta[u]) for u in pre]) if pre else 0.0
        return self.theta[v] - self.cfg.prereq_penalty * deficit

    def p_correct(self, v: int, b: float, adapted: bool = False) -> float:
        b = b + (self.cfg.adapted_difficulty_shift if adapted else 0.0)
        c = self.cfg
        return c.guess + (1 - c.guess - c.slip) * sigmoid(c.slope * (self.effective_theta(v) - b))

    def true_mastery(self) -> np.ndarray:
        return sigmoid(self.theta)

    def respond(self, v: int, b: float, adapted: bool = False) -> tuple[int, float, int]:
        """Returns (correct, latency_ms, help_requested)."""
        p = self.p_correct(v, b, adapted)
        correct = int(self.rng.random() < p)
        lat = np.exp(self.log_latency_base + self.cfg.group_latency_shift[self.group]
                     + 0.4 * (b - self.theta[v]) + self.rng.normal(0, 0.35))
        help_req = int(self.rng.random() < 0.6 * (1 - p) * (0.5 + self.frustration))
        return correct, float(lat), help_req

    def learn(self, move: Move, v: int, b: float):
        gap = b - self.theta[v]
        gain = self.lr[v] * MOVE_EFFICACY[move] * np.exp(-(gap - self.cfg.zpd_peak) ** 2 / (2 * self.cfg.zpd_width ** 2))
        gain *= (1 - 0.5 * self.frustration)
        self.theta[v] += gain
        # remediation of v also helps dependants a little via reduced deficit (already implicit)
        p = self.p_correct(v, b)
        self.frustration = float(np.clip(0.85 * self.frustration + 0.15 * (1.0 if p < 0.3 else 0.0), 0, 1))


class Population:
    """Generates simulated learners with group-conditional generative differences."""

    def __init__(self, graph: SkillGraph, params: SkillParams, cfg: SimConfig | None = None, seed: int = 0):
        self.graph, self.params, self.cfg = graph, params, cfg or SimConfig()
        self.seed = seed

    def schema_for(self, group: int, rng: np.random.Generator) -> Schema:
        vals, conf = {}, {}
        for d, options in SCHEMA_DIMS.items():
            typical = options[group % len(options)]
            vals[d] = typical if rng.random() < self.cfg.schema_group_corr else options[rng.integers(len(options))]
            conf[d] = float(rng.uniform(0.4, 1.0))
        return Schema(vals, conf)

    def sample(self, n: int) -> list[SimLearner]:
        out = []
        for i in range(n):
            rng = np.random.default_rng((self.seed, i))
            g = int(rng.integers(self.cfg.n_groups))
            theta = rng.normal(self.params.theta0_mean + self.cfg.group_theta_offset[g], self.params.theta0_sd)
            out.append(SimLearner(i, g, theta, self.schema_for(g, rng), rng, self.cfg, self.graph,
                                  self.params.lr.copy()))
        return out

    def generate_logs(self, n_learners: int, steps: int, seed: int = 1) -> pd.DataFrame:
        """Curriculum-order practice logs (for training the learner model on synthetic data)."""
        rows = []
        order = self.graph.topological_order()
        for L in self.sample(n_learners):
            rng = np.random.default_rng((seed, L.learner_id))
            pos = 0
            for t in range(steps):
                v = order[min(pos + int(rng.integers(0, 4)), len(order) - 1)]
                b = self.params.b_center[v] + rng.normal(0, 0.5)
                c, lat, h = L.respond(v, b)
                rows.append((L.learner_id, L.group, v, c, t, lat, h))
                L.learn(Move.ASSESS if rng.random() < 0.5 else Move.WORKED_EXAMPLE, v, b)
                if L.true_mastery()[v] > 0.8 or rng.random() < 0.15:
                    pos += 1
        return pd.DataFrame(rows, columns=["user", "group", "skill", "correct", "seq_idx", "latency_ms", "hints"])
