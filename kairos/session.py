"""Runs simulated learners through the full per-turn loop (Algorithm 1)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .orchestrator import Orchestrator


def run_learner(learner, policy, belief, caa, guardrail, retrieval, steps: int, adapt: bool = True,
                r_phi: int = 3, r_core: int = 3, schema=None):
    schema = schema or learner.schema
    belief.reset()
    orch = Orchestrator(policy, belief, caa, guardrail, retrieval, r_phi, r_core, adapt)
    pre = learner.true_mastery().mean()
    rows, timings = [], []
    for t in range(steps):
        s = belief.state()
        res = orch.turn(schema)
        v = res.core.skill
        item = next(it for it in retrieval.bank.by_skill[v] if it.item_id == res.item_id)
        true_m = float(learner.true_mastery()[v])
        correct, lat, help_req = learner.respond(v, item.b, adapted=res.adapted)
        learner.learn(res.core.move, v, item.b)
        belief.update(res.core.move, v, res.core.difficulty, correct, lat, help_req)
        rows.append(dict(learner=learner.learner_id, group=learner.group, t=t, skill=v,
                         depth=policy.graph.depth(v), move=res.core.move.value, difficulty=res.core.difficulty,
                         m_target=float(s.m[v]), sigma_target=float(s.sigma[v]), streak=float(s.e[0]),
                         latency_z=float(s.e[1]), help_rate=float(s.e[2]), true_m_target=true_m,
                         correct=correct, adapted=res.adapted, regenerations=res.regenerations,
                         core_rejections=res.core_rejections, fallback=res.fallback or ""))
        timings.append(res.timings_ms)
    post = learner.true_mastery().mean()
    gain = dict(learner=learner.learner_id, group=learner.group, pre=pre, post=post,
                g_norm=(post - pre) / max(1 - pre, 1e-6))
    return rows, gain, timings


def run_population(learners, make_policy, make_belief, caa, guardrail, retrieval, steps, **kw):
    rows, gains, timings = [], [], []
    for L in learners:
        r, g, tm = run_learner(L, make_policy(), make_belief(), caa, guardrail, retrieval, steps, **kw)
        rows += r; gains.append(g); timings += tm
    return pd.DataFrame(rows), pd.DataFrame(gains), pd.DataFrame(timings)
