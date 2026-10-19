"""Policy simulation for §5.7 / Table 12: ZAR, treatment-parity gap, schema-swap audit, ELG gap.

Conditions
  K               full KAIROS (core sees s_t only; latency z-scored within learner)
  K-latpop        K, but engagement latency normalised against the population (R2.1 e_t channel)
  K-FAIR          positive control: the core reads the cultural schema (simulation only, R2.6)
  K-CAA           K without the presentation transform (equivalence test on gain)
  ClassicITS      mastery-learning tutor at fixed item difficulty

  python experiments/run_policy_sim.py --n-learners 400 --steps 60
  python experiments/run_policy_sim.py --belief grkt --grkt-ckpt results/grkt.pt --n-learners 100
"""
import argparse, time
import numpy as np
import pandas as pd
from common import add_world_args, build_world, save_json, RESULTS
from kairos.simulator import Population, SimConfig
from kairos.policy import KairosPolicy, LeakyKairosPolicy, MasteryLearningPolicy, PolicyConfig
from kairos.belief import GaussianIRTBelief, GRKTBelief
from kairos.presentation import TemplateCAA, RegexClaimExtractor
from kairos.verifier import ContentVerifier, ExactClaimNLI
from kairos.orchestrator import Guardrail
from kairos.session import run_population, run_learner
from kairos.audit import tpg, tpg_permutation_test, elg_gap, zar

p = argparse.ArgumentParser()
add_world_args(p)
p.add_argument("--n-learners", type=int, default=400)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--n-min", type=int, default=20)
p.add_argument("--n-perm", type=int, default=200)
p.add_argument("--belief", choices=["gaussian", "grkt"], default="gaussian")
p.add_argument("--grkt-ckpt")
p.add_argument("--conditions", default="K,K-latpop,K-FAIR,K-CAA,ClassicITS")
p.add_argument("--swap-learners", type=int, default=100)
args = p.parse_args()

graph, params, bank, ra, logs = build_world(args)
pcfg = PolicyConfig()
pop = Population(graph, params, SimConfig(), seed=args.seed)
ga = Guardrail(ContentVerifier(ExactClaimNLI(), RegexClaimExtractor()))
caa = TemplateCAA()

if args.belief == "grkt":
    import torch
    from kairos.grkt import GRKT, GRKTConfig
    model = GRKT(graph, GRKTConfig()); model.load_state_dict(torch.load(args.grkt_ckpt)); model.eval()
    def belief_factory(norm="within"):
        return lambda: GRKTBelief(model, graph, latency_norm=norm)
else:
    def belief_factory(norm="within"):   # group-blind prior: population mean ability
        return lambda: GaussianIRTBelief(graph, params.theta0_mean, lr_hat=params.lr, latency_norm=norm)

CONDS = {
    "K": (lambda: KairosPolicy(graph, ra, pcfg), belief_factory(), True),
    "K-latpop": (lambda: KairosPolicy(graph, ra, pcfg), belief_factory("population"), True),
    "K-FAIR": (lambda: LeakyKairosPolicy(graph, ra, pcfg), belief_factory(), True),
    "K-CAA": (lambda: KairosPolicy(graph, ra, pcfg), belief_factory(), False),
    "ClassicITS": (lambda: MasteryLearningPolicy(graph, ra, pcfg), belief_factory(), False),
}

summary, tables = {}, []
for name in args.conditions.split(","):
    mk_pol, mk_bel, adapt = CONDS[name]
    t0 = time.time()
    rows, gains, _ = run_population(pop.sample(args.n_learners), mk_pol, mk_bel, caa, ga, ra, args.steps, adapt=adapt)
    pt = tpg_permutation_test(rows, args.n_min, args.n_perm, args.seed)
    tp, cov = pt["TPG"], pt["coverage"]
    z = zar(rows, pcfg.delta_lo, pcfg.delta_hi)
    e = elg_gap(gains, seed=args.seed)
    res = {**{f"TPG_{k}" if k != "TPG" else k: v for k, v in pt.items()}, **z, **e,
           "mean_gain": float(gains.g_norm.mean()), "adapted_rate": float(rows.adapted.mean()),
           "regenerations_per_turn": float(rows.regenerations.mean()),
           "neutral_fallback_rate": float((rows.fallback == "neutral_presentation").mean()),
           "safe_default_rate": float((rows.fallback == "safe_default_move").mean()),
           "seconds": round(time.time() - t0, 1)}
    summary[name] = res
    rows.assign(condition=name).to_csv(f"{RESULTS}/policy_rows_{name}.csv", index=False)
    print(f"{name:11s} TPG {tp:.4f} (null {pt['null_mean']:.4f}, p={pt['p_value']:.3f}) cov {cov:.2f} | ZAR belief {z['zar_belief']:.2f} "
          f"true {z['zar_true']:.2f} | gain {res['mean_gain']:.3f} ELG gap {e['elg_gap']:.3f} | "
          f"adapted {res['adapted_rate']:.2f}")

# ---- schema-swap audit: replay identical learners with a permuted schema ------------------
def swap_audit(name):
    mk_pol, mk_bel, adapt = CONDS[name]
    A, B = pop.sample(args.swap_learners), pop.sample(args.swap_learners)   # identical seeds
    donors = pop.sample(args.swap_learners)
    same, total = 0, 0
    for i, (a, b) in enumerate(zip(A, B)):
        other = next(d for d in donors[i + 1:] + donors[:i] if d.group != a.group)
        ra_rows, _, _ = run_learner(a, mk_pol(), mk_bel(), caa, ga, ra, args.steps, adapt=adapt)
        rb_rows, _, _ = run_learner(b, mk_pol(), mk_bel(), caa, ga, ra, args.steps, adapt=adapt, schema=other.schema)
        for x, y in zip(ra_rows, rb_rows):
            total += 1
            same += (x["move"], x["skill"], x["difficulty"]) == (y["move"], y["skill"], y["difficulty"])
    return same / total

for name in [c for c in ("K", "K-FAIR") if c in summary]:
    summary[name]["schema_swap_identical_core"] = swap_audit(name)
    print(f"schema-swap audit {name}: identical core on {summary[name]['schema_swap_identical_core']:.1%} of turns")

if "K" in summary and "K-CAA" in summary:
    summary["K_vs_K-CAA_gain_diff"] = summary["K"]["mean_gain"] - summary["K-CAA"]["mean_gain"]

summary["setup"] = vars(args) | {"n_skills": graph.n_skills, "n_prereq_edges": len(graph.prereq_edges),
                                  "sim_config": SimConfig().__dict__, "policy_config": pcfg.__dict__}
save_json("policy_sim.json", summary)
pd.DataFrame({k: v for k, v in summary.items() if isinstance(v, dict) and "TPG" in v}).T.to_csv(
    f"{RESULTS}/table12_policy_sim.csv")
print(f"wrote {RESULTS}/table12_policy_sim.csv")
