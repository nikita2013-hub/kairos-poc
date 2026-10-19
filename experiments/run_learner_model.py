"""Learner-model evaluation for §5.7 / Fig. 11 / R1.10: GRKT vs AttnKT (= K-KG ablation) vs DKT.

Reports test AUC, ECE (before and after temperature scaling), prerequisite consistency, and AUC
on prerequisite-skipping interactions; writes a reliability diagram.

  python experiments/run_learner_model.py                       # synthetic smoke run
  python experiments/run_learner_model.py --assistments skill_builder_data.csv --epochs 20
"""
import argparse, os
import numpy as np
import torch
from common import add_world_args, build_world, save_json, RESULTS
from kairos.simulator import Population, SimConfig
from kairos.data import to_sequences, split_by_learner
from kairos.grkt import GRKT, GRKTConfig, DKT, train_model, fit_temperature, evaluate, collect_logits

p = argparse.ArgumentParser()
add_world_args(p)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--n-learners", type=int, default=800, help="synthetic only")
p.add_argument("--steps", type=int, default=80, help="synthetic only")
p.add_argument("--max-len", type=int, default=200)
args = p.parse_args()
torch.manual_seed(args.seed)

graph, params, bank, ra, logs = build_world(args)
if logs is None:
    logs = Population(graph, params, SimConfig(), seed=args.seed).generate_logs(args.n_learners, args.steps)
tr, va, te = (to_sequences(d, args.max_len) for d in split_by_learner(logs, seed=args.seed))
print(f"{graph.n_skills} skills, {len(graph.prereq_edges)} prerequisite edges; "
      f"sequences train/val/test = {len(tr)}/{len(va)}/{len(te)}")

models = {
    "GRKT": GRKT(graph, GRKTConfig(max_len=args.max_len)),
    "AttnKT (K-KG)": GRKT(graph, GRKTConfig(max_len=args.max_len, use_graph=False,
                                            lambda_prereq=0, lambda_smooth=0, lambda_calib=0)),
    "DKT": DKT(graph.n_skills),
}
results, curves = {}, {}
for name, m in models.items():
    print(f"training {name}")
    train_model(m, tr, va, epochs=args.epochs, seed=args.seed)
    raw = evaluate(m, te, graph)
    T = fit_temperature(m, va)
    cal = evaluate(m, te, graph)
    results[name] = {"uncalibrated": raw, "temperature": T, "calibrated": cal}
    z, y = collect_logits(m, te)
    curves[name] = (torch.sigmoid(z / m.log_T.exp()).numpy(), y.numpy())
    print(f"  {name}: AUC {cal['auc']:.4f}  ECE {raw['ece']:.4f}->{cal['ece']:.4f}  "
          f"prereq-consistency {cal.get('prereq_consistency', float('nan')):.3f}  "
          f"AUC(prereq-skip) {cal.get('auc_prereq_skip', float('nan')):.4f}")
    torch.save(m.state_dict(), os.path.join(RESULTS, f"{name.split()[0].lower()}.pt"))

results["setup"] = {"data": args.assistments or "synthetic", "n_skills": graph.n_skills,
                    "n_prereq_edges": len(graph.prereq_edges), "epochs": args.epochs, "seed": args.seed}
save_json("learner_model.json", results)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
for name, (pp, yy) in curves.items():
    bins = np.minimum((pp * 10).astype(int), 9)
    xs = [pp[bins == b].mean() for b in range(10) if (bins == b).sum() > 20]
    ys = [yy[bins == b].mean() for b in range(10) if (bins == b).sum() > 20]
    ax.plot(xs, ys, "o-", ms=3, label=f"{name} (ECE {results[name]['calibrated']['ece']:.3f})")
ax.set_xlabel("Predicted probability correct"); ax.set_ylabel("Observed frequency correct")
ax.set_title("Learner-model reliability (test set)"); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig11_reliability.png"), dpi=200)
print("wrote results/fig11_reliability.png")
