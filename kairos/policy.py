"""Pedagogy Agent: the Kairos-score policy (manuscript §4.3).

The constrained POMDP is the specification; this module is the deployed one-step heuristic.
decide() receives only the pedagogical state s_t (no cultural schema) and a read-only view of
the belief model used for look-ahead. Closed forms (R2.5):
  p     = p(correct | v, d)
  dm    = p * (m+ - m) + (1 - p) * (m- - m)          one-step look-ahead through the transition
  Z     = triangular in [m + d_lo, m + d_hi]
  U     = info(move) * (sigma - (p * sigma+ + (1 - p) * sigma-))
  F     = sigmoid(w . [streak, latency_z, help_rate, d - m] + b)
  K     = dm * Z + omega * U - rho * F
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

from .graph import SkillGraph
from .content import RetrievalAgent
from .types import Core, Move, MOVE_INFO, PedState


@dataclass
class PolicyConfig:
    theta: float = 0.85
    delta_lo: float = 0.05
    delta_hi: float = 0.25
    omega: float = 0.5
    rho: float = 0.7
    frust_w: tuple[float, float, float, float] = (1.2, 0.6, 1.0, 2.0)
    frust_b: float = -2.0
    safe_max_frustration: float = 0.85
    pivotal_sigma: float = 0.08
    max_targets: int = 6


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@dataclass
class Scored:
    core: Core
    item_id: str
    K: float
    factors: dict
    feasible: bool
    reasons: list = field(default_factory=list)


class KairosPolicy:
    name = "K"

    def __init__(self, graph: SkillGraph, retrieval: RetrievalAgent, cfg: PolicyConfig = PolicyConfig()):
        self.graph, self.ra, self.cfg = graph, retrieval, cfg

    # ---- move availability (Table 4 triggers) ------------------------------
    def moves_for(self, s: PedState, v: int) -> list[Move]:
        m, streak = s.m[v], s.e[0]
        moves = [Move.ELICIT]
        if m < 0.5: moves.append(Move.WORKED_EXAMPLE)
        if m < 0.6: moves.append(Move.ANALOGY)
        if streak > 0: moves.append(Move.HINT)
        if streak >= 0.4: moves.append(Move.EXPLANATION)
        if streak == 0 and m >= 0.5: moves.append(Move.REFLECT)
        if m >= self.cfg.theta - 0.15: moves.append(Move.ASSESS)
        return moves

    def targets(self, s: PedState) -> list[tuple[int, Move | None]]:
        cfg = self.cfg
        F = self.graph.frontier(s.m, cfg.theta)
        if not F:   # everything mastered or blocked: consolidate the least-mastered skill
            F = [int(np.argmin(s.m))]
        F = sorted(F, key=lambda v: (self.graph.depth(v), -s.sigma[v]))[:cfg.max_targets]
        out = [(v, None) for v in F]
        for v in F:   # borderline prerequisites can be remediated
            for u in self.graph.prereqs(v):
                if s.m[u] < cfg.theta + 0.05:
                    out.append((u, Move.REMEDIATE))
        return out

    def band(self, m: float) -> tuple[float, float]:
        return min(m + self.cfg.delta_lo, 0.97), min(m + self.cfg.delta_hi, 0.99)

    # ---- scoring ------------------------------------------------------------
    def score(self, s: PedState, belief, move: Move, v: int, exclude_items=frozenset()) -> Scored:
        cfg = self.cfg
        m = float(s.m[v]); lo, hi = self.band(m)
        item = self.ra.fetch(v, (lo + hi) / 2, exclude_items)
        d = self.ra.bank.difficulty(item)
        p = belief.p_correct(v, d)
        mp, sp = belief.peek(move, v, d, 1)
        mn, sn = belief.peek(move, v, d, 0)
        dm = p * (mp - m) + (1 - p) * (mn - m)
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        Z = max(0.0, 1 - abs(d - mid) / half) if lo <= d <= hi else 0.0
        U = MOVE_INFO[move] * (float(s.sigma[v]) - (p * sp + (1 - p) * sn))
        w = cfg.frust_w
        Fr = sigmoid(w[0] * s.e[0] * 5 / 5 + w[1] * s.e[1] + w[2] * s.e[2] + w[3] * (d - m) + cfg.frust_b)
        K = dm * Z + cfg.omega * U - cfg.rho * Fr
        reasons = []
        if not (lo <= d <= hi): reasons.append("ZPD")
        low_pressure = move in (Move.WORKED_EXAMPLE, Move.HINT, Move.EXPLANATION, Move.ANALOGY)
        if Fr > cfg.safe_max_frustration and not low_pressure: reasons.append("SAFE")
        return Scored(Core(move, v, round(d, 4)), item.item_id, K,
                      dict(p=p, dm=dm, Z=Z, U=U, F=Fr, m=m, sigma=float(s.sigma[v]), band=(lo, hi)),
                      not reasons, reasons)

    def rank(self, s: PedState, belief, exclude_items=frozenset()) -> list[Scored]:
        """All candidates, feasible ones first, each group sorted by K (descending)."""
        assert isinstance(s, PedState), "Pedagogy Agent accepts only the pedagogical state"
        cands = []
        for v, forced in self.targets(s):
            for mv in ([forced] if forced else self.moves_for(s, v)):
                cands.append(self.score(s, belief, mv, v, exclude_items))
        return sorted(cands, key=lambda c: (not c.feasible, -c.K))


class LeakyKairosPolicy(KairosPolicy):
    """POSITIVE CONTROL for the audit (simulation only; = K-FAIR in the revised Table 9).
    Reads the cultural schema and lowers the difficulty band for some schema values -- exactly
    the 'responsiveness leaking into expectation' failure the architecture is designed to prevent."""
    name = "K-FAIR(leaky)"
    lowered = {("language_register", "hinglish"), ("referential_context", "rural_farming")}

    def __init__(self, *a, shift: float = 0.12, **k):
        super().__init__(*a, **k)
        self.shift = shift
        self._schema = None

    def set_schema(self, schema):     # the leak: the PA is handed c_t
        self._schema = schema

    def band(self, m):
        lo, hi = super().band(m)
        if self._schema and any((k, v) in self.lowered for k, v in self._schema.values.items()):
            return max(lo - self.shift, 0.01), max(hi - self.shift, 0.05)
        return lo, hi


class MasteryLearningPolicy:
    """Classic-ITS baseline: practise the first frontier skill at a fixed item difficulty
    (median of the skill's bank) until estimated mastery >= theta (Corbett & Anderson style)."""
    name = "ClassicITS"

    def __init__(self, graph: SkillGraph, retrieval: RetrievalAgent, cfg: PolicyConfig = PolicyConfig()):
        self.graph, self.ra, self.cfg = graph, retrieval, cfg

    def rank(self, s: PedState, belief, exclude_items=frozenset()) -> list[Scored]:
        F = self.graph.frontier(s.m, self.cfg.theta) or [int(np.argmin(s.m))]
        v = min(F, key=self.graph.depth)
        items = self.ra.bank.by_skill[v]
        med = float(np.median([self.ra.bank.difficulty(i) for i in items]))
        item = self.ra.fetch(v, med, exclude_items)
        d = self.ra.bank.difficulty(item)
        return [Scored(Core(Move.ASSESS, v, round(d, 4)), item.item_id, 0.0, dict(m=float(s.m[v])), True)]
