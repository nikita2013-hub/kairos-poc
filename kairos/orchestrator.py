"""Blackboard and coordination protocol (manuscript §4.1.3, Algorithm 1).

* Sequential per-turn execution (no intra-turn concurrency -> no races).
* Single writer per blackboard slot, enforced.
* Interface check: the Pedagogy Agent is called with (PedState, belief view) only.
* Presentation rejection -> up to R_phi regenerations with decreasing latitude, then the
  neutral y_ped is emitted unadapted.  Core rejection -> next-ranked feasible core, up to
  R_core times, then a safe default move (ELICIT on the target at the lower ZPD edge).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time

from .types import Core, Move, PedState, Schema
from .explain import provenance_trace, gloss

OWNERS = {"belief": "LMA", "evidence": "RA", "core": "PA", "presentation": "CAA",
          "trace": "EA", "verdict": "GA"}

SAFETY_BLOCKLIST = ("stupid", "lazy", "people like you", "your kind")


class Blackboard:
    def __init__(self):
        self.slots = {}

    def write(self, slot: str, value, writer: str):
        if OWNERS[slot] != writer:
            raise PermissionError(f"{writer} may not write slot '{slot}' (owner: {OWNERS[slot]})")
        self.slots[slot] = value

    def read(self, slot):
        return self.slots.get(slot)


@dataclass
class TurnResult:
    core: Core
    item_id: str
    text: str
    trace: dict
    gloss: str
    adapted: bool
    regenerations: int
    core_rejections: int
    fallback: str | None
    timings_ms: dict = field(default_factory=dict)


class Guardrail:
    def __init__(self, verifier, policy_cls_allows_schema=False):
        self.verifier = verifier

    def check_core(self, scored) -> list:
        return [] if scored.feasible else list(scored.reasons)

    def check_presentation(self, neutral, adapted, claims) -> dict:
        v = self.verifier.verify(neutral, adapted, claims)
        reasons = list(v.reasons)
        if any(b in adapted.lower() for b in SAFETY_BLOCKLIST):
            reasons.append("safety")
        return {"ok": not reasons, "reasons": reasons, "fwd": v.forward, "bwd": v.backward,
                "load_delta": round(v.load_delta, 3)}


class Orchestrator:
    def __init__(self, policy, belief, caa, guardrail: Guardrail, retrieval, r_phi: int = 3, r_core: int = 3,
                 adapt: bool = True):
        self.policy, self.belief, self.caa, self.ga, self.ra = policy, belief, caa, guardrail, retrieval
        self.r_phi, self.r_core, self.adapt = r_phi, r_core, adapt
        self.recent: list[str] = []

    def turn(self, schema: Schema) -> TurnResult:
        bb, tm = Blackboard(), {}
        t0 = time.perf_counter()
        s: PedState = self.belief.state()
        bb.write("belief", s, "LMA")
        tm["belief"] = (time.perf_counter() - t0) * 1e3

        # ---- Pedagogy Agent: receives s_t and a belief view only (interface check) ----
        t0 = time.perf_counter()
        if hasattr(self.policy, "set_schema"):     # only the leaky positive control has this
            self.policy.set_schema(schema)
        ranked = self.policy.rank(s, self.belief, frozenset(self.recent[-10:]))
        tm["policy"] = (time.perf_counter() - t0) * 1e3

        core_rej, fallback, verdicts = 0, None, []
        chosen = None
        for cand in ranked[: self.r_core]:
            reasons = self.ga.check_core(cand)
            if not reasons:
                chosen = cand
                break
            core_rej += 1
            verdicts.append({"core": cand.core.move.value, "rejected": reasons})
        if chosen is None:   # safe default move
            v = ranked[0].core.skill if ranked else 0
            lo, _ = self.policy.band(float(s.m[v]))
            item = self.ra.fetch(v, lo)
            from .policy import Scored
            chosen = Scored(Core(Move.ELICIT, v, round(self.ra.bank.difficulty(item), 4)), item.item_id, 0.0,
                            {"m": float(s.m[v])}, True)
            fallback = "safe_default_move"
        bb.write("core", chosen.core, "PA")

        item = next(it for it in self.ra.bank.by_skill[chosen.core.skill] if it.item_id == chosen.item_id)
        bb.write("evidence", item, "RA")

        # ---- presentation + content invariance ----
        regen, text, adapted, ops = 0, item.stem, False, []
        t_caa = t_ver = 0.0
        if self.adapt:
            for k in range(self.r_phi):
                lat = 1.0 - k / self.r_phi
                t1 = time.perf_counter()
                cand_text = self.caa.adapt(item, schema, lat)
                t_caa += time.perf_counter() - t1
                t1 = time.perf_counter()
                v = self.ga.check_presentation(item.stem, cand_text, item.claims)
                t_ver += time.perf_counter() - t1
                verdicts.append({"latitude": round(lat, 2), **v})
                if v["ok"]:
                    text, adapted, ops = cand_text, True, list(schema.strong().keys())
                    break
                regen += 1
            if not adapted:
                fallback = fallback or "neutral_presentation"
        tm["presentation"], tm["verifier"] = t_caa * 1e3, t_ver * 1e3
        bb.write("presentation", text, "CAA")
        bb.write("verdict", verdicts, "GA")

        t0 = time.perf_counter()
        tr = provenance_trace(chosen, ranked, s, item, verdicts, ops, fallback)
        g = gloss(tr)
        bb.write("trace", tr, "EA")
        tm["explain"] = (time.perf_counter() - t0) * 1e3
        self.recent.append(item.item_id)
        return TurnResult(chosen.core, item.item_id, text, tr, g, adapted, regen, core_rej, fallback, tm)
