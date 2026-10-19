# KAIROS proof-of-concept

Reference implementation for the revised manuscript *A Neurosymbolic Multiagent Framework for
Culturally Responsive, Personalised, Explainable and Fair AI Tutoring* (Discover AI, d57e048d).
It instantiates the full per-turn loop of Algorithm 1 and produces the numbers the response
letter promises for §5.7, Table 6, Table 12 and Fig. 11.

## Layout

| Module | Manuscript | What it does |
|---|---|---|
| `kairos/graph.py` | §4.1.2 | Prerequisite DAG, mastery frontier, heuristic graph inference for logs without an expert graph |
| `kairos/grkt.py` | §4.1.1 | GRKT (causal attentive encoder + GCN, corrected hinge, soft-binned ECE), AttnKT (= K−KG), DKT; temperature scaling; AUC, ECE, prerequisite consistency, prerequisite-skip AUC |
| `kairos/belief.py` | §4.1 | Learner-Model Agent: `GRKTBelief` (MC-dropout σ) and a fast `GaussianIRTBelief`; behavioural engagement e_t |
| `kairos/types.py` | §3.2 | `PedState` (m, σ, e) — no schema field — and `Core` |
| `kairos/policy.py` | §4.3 | Kairos score with closed-form Δm, Z, U, F̂; `LeakyKairosPolicy` positive control; classic mastery-learning ITS |
| `kairos/presentation.py` | §4.2 | Cultural Adaptation Agent (template or Anthropic LLM), claim extraction, linguistic load |
| `kairos/verifier.py` | §4.2.2 | Bidirectional NLI on extracted claims + load bound (exact-match stub or Hugging Face NLI) |
| `kairos/orchestrator.py` | §4.1.3 | Blackboard with single-writer slots, Guardrail, retry budgets R_Φ/R_core, safe default move |
| `kairos/explain.py` | §4.5 | Provenance trace, exact counterfactual explanation, templated gloss |
| `kairos/audit.py` | §4.6, §5.2 | TPG (coarsened exact matching + Jensen–Shannon) with permutation null, ELG gap, ZAR |
| `kairos/simulator.py` | §5.1 | Simulated learners with group-conditional generative differences; calibration from logs |

## Install and test

```bash
pip install -r requirements.txt
python -m pytest -q tests        # 9 tests; the hinge test needs torch
```

## Running it

```bash
./run_all.sh                                   # synthetic data end to end
./run_all.sh path/to/skill_builder_data.csv    # ASSISTments 2009-2010 skill-builder
```

Or step by step:

1. **Learner model** (fills R1.1 item 1, R1.10, Fig. 11): `python experiments/run_learner_model.py --assistments FILE --n-skills 100 --epochs 20`
2. **Policy simulation** (R1.1 items 2–3, R2.1, Table 12): `python experiments/run_policy_sim.py --assistments FILE --n-skills 100`
3. **Verifier** (R1.1 item 4, R1.4): `run_verifier_eval.py generate`, human annotation, then `evaluate`
4. **Latency** (R1.11, R3.2, Table 6, Fig. 8): `python experiments/run_latency.py` with real backends

All outputs go to `results/`.

## Which letter placeholder each output fills

| Letter item | Source |
|---|---|
| R1.1 dataset, N learners, LLM, NLI model | the arguments you run with |
| R1.1 item 1, Fig. 11 | `results/learner_model.json`, `results/fig11_reliability.png` |
| R1.1 items 2–3, R2.1 schema-swap result, Table 12 | `results/policy_sim.json`, `results/table12_policy_sim.csv` |
| R1.1 item 4, R1.4 rates | `results/verifier_eval.json` (human labels only) |
| R1.10 prerequisite-skip AUC | `auc_prereq_skip` in `learner_model.json` |
| R1.11 / R3.2 Table 6 | `results/table6_latency.csv`, `results/latency.json` |
| R2.5 θ, Table 10 | `PolicyConfig` and `GRKTConfig` defaults (θ = 0.85) |

## What has and has not been run

Everything except the learner models has been executed. In the sandbox where this code was written, the
numpy components, all experiment scripts and 8 of 9 unit tests ran successfully on synthetic data.
**`kairos/grkt.py` and `GRKTBelief` were only compiled and statically checked there**, because PyTorch
could not be installed. Run `pytest` and the learner-model script first on your machine and fix anything
before relying on the numbers.

## Things the paper and the letter must state honestly

1. **Simulation cannot test presentation effects on learning.** Simulated responses do not depend on
   framing (`adapted_difficulty_shift = 0`), so K and K−CAA have identical gains by construction. Report
   that comparison as a sanity check only; the equivalence test belongs to the human study.
2. **The exact-match NLI and regex extractor are an oracle for the templated domain.** Their FAR is 0 by
   construction. Publishable FRR/FAR need the HF NLI model, LLM adaptations and human labels.
3. **The linguistic-load bound drives most rejections.** With template adaptations and η_L = 0.15,
   context sentences raise the load proxy on short stems. Report η_L and the rejection breakdown, and
   consider a sensitivity analysis.
4. **TPG uses a permutation null, not a bootstrap.** Plug-in Jensen–Shannon is biased upward in small
   strata, so a group-blind policy shows TPG > 0. Report TPG with the permutation-null mean and p-value.
   The letter (R2.2, "Inference") says "learner-level bootstrap" and should be updated to match.
5. **The schema-swap audit is a positive/negative control pair.** It shows 100% identical cores for K, and
   a large difference for the leaky K−FAIR control. This validates the audit's power; it is not evidence
   about real learners.
6. **ZPD adherence is reported against belief and true mastery.** Belief-relative adherence is near 1 by
   design; true-mastery adherence measures how well the learner model tracks the learner.
7. **The engagement channel is real.** With population-normalised latency (K-latpop), group latency shifts
   enter e_t. Within-learner normalisation is the design choice that closes it; report both.
8. **No PPO-Lagrangian comparison is included.** Reword R3.3 item 3 in the letter as planned work, or add it.
9. **The ASSISTments graph is inferred.** `SkillGraph.infer_from_logs` is a heuristic stand-in. Say so, or
   supply an expert graph with `--graph`.
10. **Move efficacies and informativeness** (`types.py`) are modelling assumptions; list them in Table 10.
