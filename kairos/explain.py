"""Explanation Agent: provenance trace, exact counterfactual, templated gloss (manuscript §4.5)."""
from __future__ import annotations
import copy
import numpy as np

from .types import PedState


def provenance_trace(chosen, ranked, s: PedState, item, verdicts, presentation_ops, fallback) -> dict:
    v = chosen.core.skill
    return {
        "target_skill": v, "mastery": round(float(s.m[v]), 3), "uncertainty": round(float(s.sigma[v]), 3),
        "core": {"move": chosen.core.move.value, "difficulty": chosen.core.difficulty},
        "factors": {k: (round(x, 4) if isinstance(x, float) else x) for k, x in chosen.factors.items()},
        "alternatives": [{"move": r.core.move.value, "skill": r.core.skill, "K": round(r.K, 4),
                          "feasible": r.feasible, "reasons": r.reasons} for r in ranked[:5]],
        "evidence": {"item_id": item.item_id, "graph_node": item.skill},
        "presentation_ops": presentation_ops, "guardrail": verdicts, "fallback": fallback,
    }


def counterfactual(policy, belief, s: PedState, chosen, max_shift: float = 0.3, tol: float = 0.005):
    """Smallest shift of the target skill's mastery that changes the chosen move (exact under
    the closed-form score: re-runs the policy on a perturbed copy of the state)."""
    v = chosen.core.skill

    def decision(shift):
        m = s.m.copy(); m[v] = float(np.clip(m[v] + shift, 0.001, 0.999))
        s2 = PedState(m=m, sigma=s.sigma, e=s.e)
        b2 = copy.deepcopy(belief)
        if hasattr(b2, "mu"):
            from .belief import logit
            b2.mu[v] = logit(m[v]) / b2._kappa(b2.var[v])
        top = policy.rank(s2, b2)[0]
        return (top.core.move, top.core.skill)

    base = (chosen.core.move, chosen.core.skill)
    best = None
    for sign in (+1, -1):
        lo, hi = 0.0, max_shift
        if decision(sign * hi) == base:
            continue
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if decision(sign * mid) == base: lo = mid
            else: hi = mid
        if best is None or hi < abs(best[0]):
            best = (sign * hi, decision(sign * hi))
    if best is None:
        return None
    return {"mastery_shift": round(best[0], 3), "alternative_move": best[1][0].value, "alternative_skill": best[1][1]}


GLOSS = {
    "worked_example": "Here's a step-by-step example on {skill}, because this skill is still new for you (about {pct}% secure).",
    "assess": "Let's check {skill}: you look close to mastering it ({pct}%).",
    "elicit": "A quick question on {skill} to see where you are; I'm not yet sure (about {pct}%).",
    "hint": "A small hint on {skill}, since the last answer didn't work out.",
    "explanation": "Let's look at the idea behind {skill} again; a couple of answers suggest a mix-up.",
    "analogy": "Here's a comparison to help connect {skill} to something familiar.",
    "reflect": "Nice work on {skill}. Can you explain how you got it?",
    "remediate_prereq": "Let's strengthen {skill} first, because the next topic builds on it.",
}


def gloss(trace: dict, skill_names=None) -> str:
    """Learner-facing gloss generated ONLY from trace slots (an LLM may paraphrase it; the
    Guardrail then checks the paraphrase against these slots)."""
    name = skill_names[trace["target_skill"]] if skill_names else f"skill {trace['target_skill']}"
    return GLOSS[trace["core"]["move"]].format(skill=name, pct=int(round(100 * trace["mastery"])))
