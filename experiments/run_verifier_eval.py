"""Content-invariance verifier evaluation for R1.4 / §5.7: false-rejection and false-acceptance.

Step 1 - generate candidate pairs for human annotation:
  python experiments/run_verifier_eval.py generate --n 300 --caa template|anthropic
  -> results/verifier_pairs.csv with label_auto (construction label) and an empty label_human.
     Negatives are built by perturbation: changed quantity, easier numbers (softening),
     added hint, dropped task. Positives are CAA adaptations (equivalent by intent; humans confirm).
Step 2 - two annotators fill label_human (1 = same content and difficulty, 0 = not); resolve.
Step 3 - evaluate:
  python experiments/run_verifier_eval.py evaluate --nli exact|hf:microsoft/deberta-large-mnli \
         --extractor regex|anthropic
FRR = rejected among label 1; FAR = accepted among label 0 (bootstrap 95% CI).
Uses label_human where present; otherwise label_auto with a warning (not publishable).
"""
import argparse, re
import numpy as np
import pandas as pd
from common import add_world_args, build_world, save_json, RESULTS
from kairos.simulator import Population, SimConfig
from kairos.presentation import TemplateCAA, RegexClaimExtractor
from kairos.verifier import ContentVerifier, ExactClaimNLI

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["generate", "evaluate"])
add_world_args(p)
p.add_argument("--n", type=int, default=300)
p.add_argument("--caa", default="template")
p.add_argument("--nli", default="exact")
p.add_argument("--extractor", default="regex")
p.add_argument("--eta-load", type=float, default=0.15)
p.add_argument("--pairs", default=f"{RESULTS}/verifier_pairs.csv")
args = p.parse_args()


def perturb(text, rng):
    nums = re.findall(r"\d+", text)
    kind = rng.choice(["change_quantity", "soften", "add_hint", "drop_task"])
    if kind == "change_quantity" and nums:
        n = rng.choice(nums); return kind, text.replace(n, str(int(n) + int(rng.integers(1, 5))), 1)
    if kind == "soften" and nums:
        n = max(nums, key=int); return kind, text.replace(n, str(max(2, int(n) // 2)), 1)
    if kind == "add_hint":
        return kind, text + " Hint: first, divide by the bottom number."
    sents = [s for s in re.split(r"(?<=[.?])\s+", text) if s]
    keep = [s for s in sents if not re.search(r"\d", s)] or ["Think about this situation."]
    return "drop_task", " ".join(keep)


if args.mode == "generate":
    graph, params, bank, ra, _ = build_world(args)
    learners = Population(graph, params, SimConfig(), seed=args.seed).sample(50)
    if args.caa == "anthropic":
        from kairos.presentation import AnthropicCAA; caa = AnthropicCAA()
    else:
        caa = TemplateCAA()
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n):
        item = bank.items[int(rng.integers(len(bank.items)))]
        L = learners[int(rng.integers(len(learners)))]
        adapted = caa.adapt(item, L.schema, float(rng.choice([1.0, 0.67, 0.33])))
        if rng.random() < 0.5:
            rows.append(dict(pair_id=i, item_id=item.item_id, neutral=item.stem, adapted=adapted,
                             construction="caa", label_auto=1, label_human=""))
        else:
            kind, bad = perturb(adapted, rng)
            rows.append(dict(pair_id=i, item_id=item.item_id, neutral=item.stem, adapted=bad,
                             construction=kind, label_auto=0, label_human=""))
    pd.DataFrame(rows).to_csv(args.pairs, index=False)
    print(f"wrote {args.pairs}: {len(rows)} pairs; ask two annotators to fill label_human")
else:
    df = pd.read_csv(args.pairs)
    human = df.label_human.notna() & (df.label_human.astype(str).str.strip() != "")
    if human.all():
        df["label"] = df.label_human.astype(int); source = "human"
    else:
        print("WARNING: label_human missing; using construction labels (smoke test only, not publishable)")
        df["label"] = np.where(human, df.label_human, df.label_auto).astype(int); source = "auto"
    nli = ExactClaimNLI()
    if args.nli.startswith("hf:"):
        from kairos.verifier import HFNLI; nli = HFNLI(args.nli[3:])
    ext = RegexClaimExtractor()
    if args.extractor == "anthropic":
        from kairos.presentation import AnthropicClaimExtractor; ext = AnthropicClaimExtractor()
    ver = ContentVerifier(nli, ext, eta_load=args.eta_load)
    v = [ver.verify(n, a) for n, a in zip(df.neutral, df.adapted)]
    df["accepted"] = [x.ok for x in v]
    df["reasons"] = [";".join(x.reasons) for x in v]
    df.to_csv(f"{RESULTS}/verifier_scored.csv", index=False)

    def rates(d):
        pos, neg = d[d.label == 1], d[d.label == 0]
        return (1 - pos.accepted.mean()) if len(pos) else np.nan, neg.accepted.mean() if len(neg) else np.nan
    frr, far = rates(df)
    rng = np.random.default_rng(0)
    boots = np.array([rates(df.sample(len(df), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)])
    out = {"label_source": source, "n": len(df), "n_pos": int((df.label == 1).sum()), "n_neg": int((df.label == 0).sum()),
           "FRR": frr, "FRR_CI": np.nanpercentile(boots[:, 0], [2.5, 97.5]).tolist(),
           "FAR": far, "FAR_CI": np.nanpercentile(boots[:, 1], [2.5, 97.5]).tolist(),
           "false_rejection_reasons": df[(df.label == 1) & ~df.accepted].reasons.value_counts().to_dict(),
           "false_accepts_by_construction": df[(df.label == 0) & df.accepted].construction.value_counts().to_dict(),
           "nli": args.nli, "extractor": args.extractor, "eta_load": args.eta_load}
    print(f"FRR {frr:.3f} {out['FRR_CI']}  FAR {far:.3f} {out['FAR_CI']}  (labels: {source})")
    save_json("verifier_eval.json", out)
