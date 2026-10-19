"""Learner-Model Agent belief adapters (manuscript §4.1).

Both adapters expose the same interface and produce a PedState s_t = (m, sigma, e):
  * GaussianIRTBelief - fast analytic belief (per-skill Gaussian on ability, Laplace update,
    learning-drift transition, graph propagation). Used for large simulation sweeps.
  * GRKTBelief        - wraps a trained GRKT model; uncertainty via MC dropout.
Engagement e_t uses behavioural signals only (R2.1): error streak, latency z-score, help rate.
Latency is z-scored WITHIN learner by default, which removes group-level baseline shifts;
`latency_norm="population"` reproduces the leaky alternative for the audit's positive control.
"""
from __future__ import annotations
import math
import numpy as np

from .graph import SkillGraph
from .types import PedState, Move, MOVE_EFFICACY

SLOPE = 1.7


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Engagement:
    def __init__(self, latency_norm: str = "within", pop_mean: float = 9.5, pop_sd: float = 0.5):
        self.latency_norm, self.pop_mean, self.pop_sd = latency_norm, pop_mean, pop_sd
        self.streak, self.help, self.n = 0, 0.0, 0
        self.lat_mean, self.lat_m2 = 0.0, 0.0
        self.last_z = 0.0

    def update(self, correct: int, latency_ms: float, help_req: int):
        self.streak = 0 if correct else min(self.streak + 1, 5)
        self.help = 0.8 * self.help + 0.2 * help_req
        x = math.log(max(latency_ms, 1.0))
        self.n += 1
        d = x - self.lat_mean
        self.lat_mean += d / self.n
        self.lat_m2 += d * (x - self.lat_mean)
        if self.latency_norm == "within":
            sd = math.sqrt(self.lat_m2 / self.n) if self.n > 2 else 1.0
            self.last_z = (x - self.lat_mean) / max(sd, 0.1) if self.n > 2 else 0.0
        else:
            self.last_z = (x - self.pop_mean) / self.pop_sd

    def vector(self) -> np.ndarray:
        return np.array([self.streak / 5.0, float(np.clip(self.last_z, -3, 3)), self.help])


class GaussianIRTBelief:
    def __init__(self, graph: SkillGraph, prior_mean: np.ndarray, prior_sd: float = 1.0,
                 lr_hat: np.ndarray | None = None, process_noise: float = 0.02,
                 spill: float = 0.25, latency_norm: str = "within"):
        self.graph = graph
        self.V = graph.n_skills
        self.prior_mean, self.prior_sd = np.asarray(prior_mean, float), prior_sd
        self.lr_hat = np.full(self.V, 0.2) if lr_hat is None else np.asarray(lr_hat, float)
        self.q, self.spill, self.latency_norm = process_noise, spill, latency_norm
        self.children = [[] for _ in range(self.V)]
        for u, v in graph.prereq_edges:
            self.children[u].append(v)
        self.reset()

    def reset(self):
        self.mu = self.prior_mean.copy()
        self.var = np.full(self.V, self.prior_sd ** 2)
        self.eng = Engagement(self.latency_norm)

    # --- model ---------------------------------------------------------------
    def _kappa(self, var):
        return 1.0 / np.sqrt(1.0 + math.pi * SLOPE ** 2 * var / 8.0)

    def p_correct(self, v: int, difficulty: float) -> float:
        b = logit(difficulty)
        return float(sig(self._kappa(self.var[v]) * SLOPE * (self.mu[v] - b)))

    def _drift(self, move: Move, v: int, difficulty: float, mu: float) -> float:
        gap = logit(difficulty) - mu
        return self.lr_hat[v] * MOVE_EFFICACY[move] * math.exp(-(gap - 0.5) ** 2 / (2 * 0.9 ** 2))

    def _observe(self, mu, var, v, difficulty, correct):
        b = logit(difficulty)
        p = sig(self._kappa(var) * SLOPE * (mu - b))
        new_var = 1.0 / (1.0 / var + SLOPE ** 2 * p * (1 - p))
        new_mu = mu + new_var * SLOPE * (correct - p)
        return new_mu, new_var

    def update(self, move: Move, v: int, difficulty: float, correct: int, latency_ms: float, help_req: int):
        mu0 = self.mu[v]
        mu1 = mu0 + self._drift(move, v, difficulty, mu0)
        var1 = self.var[v] + self.q
        self.mu[v], self.var[v] = self._observe(mu1, var1, v, difficulty, correct)
        delta = self.mu[v] - mu0
        # graph propagation: success on v is evidence for its prerequisites; failure on v is
        # (weak) evidence against its dependants
        if delta > 0:
            for u in self.graph.prereqs(v):
                self.mu[u] += self.spill * delta
        else:
            for w in self.children[v]:
                self.mu[w] += self.spill * delta
        self.eng.update(correct, latency_ms, help_req)

    def mastery(self) -> np.ndarray:
        return sig(self._kappa(self.var) * self.mu)

    def state(self) -> PedState:
        m = self.mastery()
        return PedState(m=m, sigma=m * (1 - m) * np.sqrt(self.var), e=self.eng.vector())

    def peek(self, move: Move, v: int, difficulty: float, correct: int) -> tuple[float, float]:
        """(m_v, sigma_v) after a hypothetical transition + observation; state unchanged."""
        mu1 = self.mu[v] + self._drift(move, v, difficulty, self.mu[v])
        mu2, var2 = self._observe(mu1, self.var[v] + self.q, v, difficulty, correct)
        m2 = float(sig(self._kappa(var2) * mu2))
        return m2, float(m2 * (1 - m2) * math.sqrt(var2))


class GRKTBelief:
    """Online wrapper around a trained GRKT/DKT model. Difficulty enters only via the move's
    response model p(correct | v, d) = sigmoid(SLOPE*(logit(m_v) - logit(d)))."""

    def __init__(self, model, graph: SkillGraph, mc_samples: int = 8, latency_norm: str = "within"):
        import torch
        self.torch, self.model, self.graph, self.k = torch, model, graph, mc_samples
        self.latency_norm = latency_norm
        self.reset()

    def reset(self):
        self.S, self.C = [], []
        self.eng = Engagement(self.latency_norm)
        self._cache = None

    def _mc(self, S, C):
        torch = self.torch
        if not S:   # empty history: use model bias only
            m = torch.sigmoid(self.model.beta.detach()).numpy()
            return m, np.full_like(m, 0.15)
        s, c = torch.tensor([S]), torch.tensor([C])
        self.model.train()   # MC dropout
        with torch.no_grad():
            draws = torch.stack([torch.sigmoid(self.model.mastery_logits(s, c)[0, -1]) for _ in range(self.k)])
        self.model.eval()
        return draws.mean(0).numpy(), draws.std(0).numpy()

    def p_correct(self, v: int, difficulty: float) -> float:
        m = self.state().m[v]
        return float(sig(SLOPE * (logit(m) - logit(difficulty))))

    def update(self, move, v, difficulty, correct, latency_ms, help_req):
        self.S.append(int(v)); self.C.append(int(correct))
        cap = getattr(getattr(self.model, "cfg", None), "max_len", 200)
        self.S, self.C = self.S[-cap:], self.C[-cap:]
        self.eng.update(correct, latency_ms, help_req)
        self._cache = None

    def state(self) -> PedState:
        if self._cache is None:
            m, sd = self._mc(self.S, self.C)
            self._cache = PedState(m=m, sigma=sd, e=self.eng.vector())
        return self._cache

    def peek(self, move, v, difficulty, correct):
        m, sd = self._mc(self.S + [int(v)], self.C + [int(correct)])
        base = self.state().m[v]
        # learning implicit in the trained transition; scale by move efficacy (PoC assumption)
        return float(base + MOVE_EFFICACY[move] * (m[v] - base)), float(sd[v])
