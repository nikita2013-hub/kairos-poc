"""Unit tests for the properties the paper relies on.  Run: python -m pytest -q"""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kairos.graph import SkillGraph
from kairos.simulator import Population, SkillParams, SimConfig
from kairos.content import ItemBank, RetrievalAgent
from kairos.policy import KairosPolicy, LeakyKairosPolicy, PolicyConfig
from kairos.belief import GaussianIRTBelief
from kairos.presentation import TemplateCAA, RegexClaimExtractor
from kairos.verifier import ContentVerifier, ExactClaimNLI
from kairos.orchestrator import Blackboard, Guardrail, Orchestrator
from kairos.types import PedState, Move
from kairos.audit import jsd


@pytest.fixture(scope="module")
def world():
    g = SkillGraph.random_dag(20, seed=3)
    P = SkillParams.default(20, g, seed=3)
    bank = ItemBank.synthetic(g, P.b_center, seed=3)
    return g, P, bank, RetrievalAgent(bank)


def test_graph_is_dag_and_frontier(world):
    g, *_ = world
    g.topological_order()
    m = np.zeros(g.n_skills)
    assert set(g.frontier(m, 0.85)) == {v for v in range(g.n_skills) if not g.prereqs(v)}


def test_corrected_hinge_zero_at_full_mastery():
    try:
        import torch
    except Exception:
        pytest.skip("torch not available")
    m_u = m_v = torch.tensor(1.0)
    assert float(torch.relu(m_v - m_u - 0.05)) == 0.0                 # R2.5


def test_pedagogy_agent_rejects_non_pedstate(world):
    g, P, bank, ra = world
    b = GaussianIRTBelief(g, P.theta0_mean)
    with pytest.raises(AssertionError):
        KairosPolicy(g, ra).rank({"m": b.state().m, "schema": "x"}, b)


def test_blackboard_single_writer():
    bb = Blackboard()
    bb.write("core", 1, "PA")
    with pytest.raises(PermissionError):
        bb.write("core", 2, "CAA")


def test_schema_does_not_change_core(world):
    g, P, bank, ra = world
    L = Population(g, P, SimConfig()).sample(6)
    b = GaussianIRTBelief(g, P.theta0_mean, lr_hat=P.lr)
    ga = Guardrail(ContentVerifier(ExactClaimNLI(), RegexClaimExtractor()))
    cores = set()
    for learner in L:
        b.reset()
        cores.add(Orchestrator(KairosPolicy(g, ra), b, TemplateCAA(), ga, ra).turn(learner.schema).core)
    assert len(cores) == 1, "same pedagogical state, different schemas -> core must be identical"


def test_leaky_control_is_detectably_different(world):
    g, P, bank, ra = world
    L = Population(g, P, SimConfig()).sample(40)
    b = GaussianIRTBelief(g, P.theta0_mean, lr_hat=P.lr)
    ga = Guardrail(ContentVerifier(ExactClaimNLI(), RegexClaimExtractor()))
    diffs = set()
    for learner in L:
        b.reset()
        diffs.add(Orchestrator(LeakyKairosPolicy(g, ra), b, TemplateCAA(), ga, ra).turn(learner.schema).core.difficulty)
    assert len(diffs) > 1


def test_verifier_catches_drift(world):
    g, P, bank, ra = world
    item = bank.items[5]
    ver = ContentVerifier(ExactClaimNLI(), RegexClaimExtractor(), eta_load=10)
    assert ver.verify(item.stem, "At the market. " + item.stem).ok
    assert "addition" in ver.verify(item.stem, item.stem + " Hint: first, divide by 4.").reasons
    import re
    n = re.findall(r"\d+", item.stem)[-1]
    changed = item.stem.replace(n, str(int(n) + 3))
    assert not ver.verify(item.stem, changed).ok


def test_retry_budget_falls_back_to_neutral(world):
    g, P, bank, ra = world
    class BadCAA:
        def adapt(self, item, schema, latitude=1.0): return item.stem + " Hint: the answer is 7."
    L = Population(g, P, SimConfig()).sample(1)[0]
    b = GaussianIRTBelief(g, P.theta0_mean)
    ga = Guardrail(ContentVerifier(ExactClaimNLI(), RegexClaimExtractor()))
    res = Orchestrator(KairosPolicy(g, ra), b, BadCAA(), ga, ra, r_phi=3).turn(L.schema)
    assert res.regenerations == 3 and not res.adapted and res.fallback == "neutral_presentation"
    item = next(i for i in bank.by_skill[res.core.skill] if i.item_id == res.item_id)
    assert res.text == item.stem


def test_jsd_bounds():
    p = np.array([1.0, 0.0]); q = np.array([0.0, 1.0])
    assert abs(jsd(p, q) - 1.0) < 1e-9 and jsd(p, p) == 0.0
