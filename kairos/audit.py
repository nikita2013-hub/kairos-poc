"""Fairness and adherence audits (manuscript §4.6, §5.2).

TPG: coarsened exact matching on s_t (target mastery deciles, uncertainty terciles,
     latency-z terciles x error-streak>0), action = move x target depth x difficulty bin (5),
     divergence = Jensen-Shannon (base 2), stratum-size weights, learner-level bootstrap.
ELG: |E[G_norm | g=a] - E[G_norm | g=a']| with bootstrap CI.
ZAR: share of actions whose difficulty lies in the proximal band, w.r.t. belief and TRUE mastery.
Schema swap: replay each learner with a permuted schema; share of turns with identical core.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _strata(df: pd.DataFrame) -> pd.Series:
    mb = pd.qcut(df.m_target.rank(method="first"), 10, labels=False)
    sb = pd.qcut(df.sigma_target.rank(method="first"), 3, labels=False)
    lb = pd.qcut(df.latency_z.rank(method="first"), 3, labels=False)
    eb = (df.streak > 0).astype(int)
    return mb.astype(str) + "_" + sb.astype(str) + "_" + lb.astype(str) + "_" + eb.astype(str)


def tpg(df: pd.DataFrame, n_min: int = 20, groups=(0, 1)) -> tuple[float, float]:
    """Returns (TPG, coverage = share of decisions in strata that met n_min per group)."""
    df = df.copy()
    df["stratum"] = _strata(df)
    df["action"] = df.move + "|" + df.depth.astype(str) + "|" + pd.cut(df.difficulty, 5, labels=False).astype(str)
    actions = sorted(df.action.unique())
    total, acc, used = 0.0, 0.0, 0
    for _, g in df.groupby("stratum"):
        a, b = g[g.group == groups[0]], g[g.group == groups[1]]
        if a.learner.nunique() < n_min or b.learner.nunique() < n_min:
            continue
        pa = a.action.value_counts(normalize=True).reindex(actions, fill_value=0).values
        pb = b.action.value_counts(normalize=True).reindex(actions, fill_value=0).values
        acc += len(g) * jsd(pa, pb); total += len(g); used += len(g)
    return (acc / total if total else float("nan")), used / len(df)


def bootstrap(fn, df: pd.DataFrame, n_boot: int = 200, seed: int = 0, by: str = "learner"):
    rng = np.random.default_rng(seed)
    ids = df[by].unique()
    groups = {k: g for k, g in df.groupby(by)}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        boot = pd.concat([groups[i].assign(**{by: f"{i}_{j}"}) for j, i in enumerate(pick)])
        vals.append(fn(boot))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def elg_gap(gains: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """gains: columns learner, group, g_norm."""
    rng = np.random.default_rng(seed)
    a = gains[gains.group == 0].g_norm.values; b = gains[gains.group == 1].g_norm.values
    est = abs(a.mean() - b.mean())
    boots = [abs(rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()) for _ in range(n_boot)]
    return {"elg_gap": float(est), "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "mean_gain_g0": float(a.mean()), "mean_gain_g1": float(b.mean())}


def zar(df: pd.DataFrame, delta_lo: float, delta_hi: float) -> dict:
    inb = (df.difficulty >= df.m_target + delta_lo - 1e-9) & (df.difficulty <= df.m_target + delta_hi + 1e-9)
    tru = (df.difficulty >= df.true_m_target + delta_lo) & (df.difficulty <= df.true_m_target + delta_hi)
    return {"zar_belief": float(inb.mean()), "zar_true": float(tru.mean())}


def tpg_permutation_test(df: pd.DataFrame, n_min: int = 20, n_perm: int = 200, seed: int = 0) -> dict:
    """Plug-in JSD is biased upward in small strata, so a group-blind policy still shows TPG > 0.
    The null distribution is obtained by permuting group labels across LEARNERS (keeping each
    learner's trajectory intact). Report TPG, the null mean / 95th percentile, the excess over
    the null mean, and a one-sided p-value."""
    rng = np.random.default_rng(seed)
    obs, cov = tpg(df, n_min)
    lg = df.groupby("learner").group.first()
    null = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(lg.values), index=lg.index)
        null.append(tpg(df.assign(group=df.learner.map(perm)), n_min)[0])
    null = np.array([x for x in null if np.isfinite(x)])
    return {"TPG": obs, "coverage": cov, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "excess_over_null": float(obs - null.mean()),
            "p_value": float((1 + (null >= obs).sum()) / (1 + len(null)))}
