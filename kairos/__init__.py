"""KAIROS proof-of-concept reference implementation.

Maps to manuscript sections:
  graph.py         §4.1.2  knowledge graph, mastery frontier
  grkt.py          §4.1.1  graph-regularised knowledge tracing (+ DKT / attentive-KT baselines)
  belief.py        §4.1    learner-model agent: belief adapters (GRKT or fast Gaussian-IRT)
  policy.py        §4.3    pedagogical state s_t, Kairos score, feasible set
  presentation.py  §4.2    cultural adaptation agent, claim extraction, linguistic load
  verifier.py      §4.2.2  bidirectional NLI content-invariance verifier
  orchestrator.py  §4.1.3  blackboard, coordination protocol, guardrail, retry budgets
  explain.py       §4.5    provenance trace, counterfactual explanation, templated gloss
  audit.py         §4.6/§5.2  TPG (CEM + JSD), ELG, ZAR, schema-swap audit
  simulator.py     §5.1    simulated learners calibrated on response logs
"""
__version__ = "0.1.0"
