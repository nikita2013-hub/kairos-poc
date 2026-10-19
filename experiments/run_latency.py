"""End-to-end per-turn latency for Table 6 / Fig. 8 / R1.11 / R3.2.

Times every stage of Algorithm 1 with the configured backends, including the presentation
transform and the verifier (and an LLM realisation of y_ped if --realise anthropic).

  python experiments/run_latency.py --turns 200                               # stubs (smoke)
  python experiments/run_latency.py --caa anthropic --nli hf:microsoft/deberta-large-mnli \
         --extractor anthropic --realise anthropic --belief grkt --grkt-ckpt results/grkt.pt
"""
import argparse, platform, time
import numpy as np
import pandas as pd
from common import add_world_args, build_world, save_json, RESULTS
from kairos.simulator import Population, SimConfig
from kairos.policy import KairosPolicy, PolicyConfig
from kairos.belief import GaussianIRTBelief, GRKTBelief
from kairos.presentation import TemplateCAA, RegexClaimExtractor
from kairos.verifier import ContentVerifier, ExactClaimNLI
from kairos.orchestrator import Guardrail, Orchestrator

p = argparse.ArgumentParser()
add_world_args(p)
p.add_argument("--turns", type=int, default=200)
p.add_argument("--learners", type=int, default=10)
p.add_argument("--caa", default="template")
p.add_argument("--nli", default="exact")
p.add_argument("--extractor", default="regex")
p.add_argument("--realise", default="bank", help="bank | anthropic (LLM realisation of y_ped)")
p.add_argument("--belief", default="gaussian")
p.add_argument("--grkt-ckpt")
args = p.parse_args()

graph, params, bank, ra, _ = build_world(args)
caa = TemplateCAA()
if args.caa == "anthropic":
    from kairos.presentation import AnthropicCAA; caa = AnthropicCAA()
nli = ExactClaimNLI()
if args.nli.startswith("hf:"):
    from kairos.verifier import HFNLI; nli = HFNLI(args.nli[3:])
ext = RegexClaimExtractor()
if args.extractor == "anthropic":
    from kairos.presentation import AnthropicClaimExtractor; ext = AnthropicClaimExtractor()
ga = Guardrail(ContentVerifier(nli, ext))
if args.belief == "grkt":
    import torch
    from kairos.grkt import GRKT, GRKTConfig
    model = GRKT(graph, GRKTConfig()); model.load_state_dict(torch.load(args.grkt_ckpt)); model.eval()
    mk_belief = lambda: GRKTBelief(model, graph)
else:
    mk_belief = lambda: GaussianIRTBelief(graph, params.theta0_mean, lr_hat=params.lr)

realiser = None
if args.realise == "anthropic":
    import anthropic
    client = anthropic.Anthropic()
    def realiser(core, item):
        client.messages.create(model="claude-haiku-4-5", max_tokens=300, messages=[{"role": "user", "content":
            f"As a tutor, produce a {core.move.value.replace('_', ' ')} for this task without solving it: {item.stem}"}])

learners = Population(graph, params, SimConfig(), seed=args.seed).sample(args.learners)
rows = []
per = max(1, args.turns // args.learners)
for L in learners:
    belief = mk_belief(); orch = Orchestrator(KairosPolicy(graph, ra), belief, caa, ga, ra)
    for _ in range(per):
        t0 = time.perf_counter()
        res = orch.turn(L.schema)
        item = next(it for it in bank.by_skill[res.core.skill] if it.item_id == res.item_id)
        t1 = time.perf_counter()
        if realiser: realiser(res.core, item)
        realise_ms = (time.perf_counter() - t1) * 1e3
        c, lat, h = L.respond(res.core.skill, item.b, res.adapted)
        L.learn(res.core.move, res.core.skill, item.b)
        t2 = time.perf_counter()
        belief.update(res.core.move, res.core.skill, res.core.difficulty, c, lat, h)
        upd_ms = (time.perf_counter() - t2) * 1e3
        rows.append({**res.timings_ms, "realise_y_ped": realise_ms, "belief_update": upd_ms,
                     "total": (t1 - t0) * 1e3 + realise_ms + upd_ms,
                     "regenerations": res.regenerations, "fallback": res.fallback or ""})
df = pd.DataFrame(rows)
stages = ["belief_update", "belief", "policy", "realise_y_ped", "presentation", "verifier", "explain", "total"]
table = pd.DataFrame({"median_ms": df[stages].median(), "p95_ms": df[stages].quantile(0.95)}).round(2)
table.to_csv(f"{RESULTS}/table6_latency.csv")
print(table)
meta = {"turns": len(df), "regeneration_rate": float((df.regenerations > 0).mean()),
        "neutral_fallback_rate": float((df.fallback == "neutral_presentation").mean()),
        "backends": {"caa": args.caa, "nli": args.nli, "extractor": args.extractor, "realise": args.realise,
                     "belief": args.belief},
        "hardware": {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()},
        "table": table.to_dict()}
save_json("latency.json", meta)
