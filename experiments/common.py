"""Shared world construction for the experiments."""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kairos.graph import SkillGraph
from kairos.simulator import SkillParams, Population, SimConfig
from kairos.content import ItemBank, RetrievalAgent
from kairos.data import load_assistments_2009

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def add_world_args(p):
    p.add_argument("--assistments", help="path to ASSISTments 2009 skill-builder CSV (default: synthetic)")
    p.add_argument("--graph", help="CSV prereq,skill (default: true DAG for synthetic; inferred for real data)")
    p.add_argument("--n-skills", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)


def build_world(args):
    """Returns graph, skill params, item bank, retrieval agent, real-log frame (or None)."""
    logs = None
    if args.assistments:
        logs, _ = load_assistments_2009(args.assistments, max_skills=args.n_skills)
        n = int(logs.skill.max()) + 1
        graph = SkillGraph.from_csv(args.graph, n) if args.graph else SkillGraph.infer_from_logs(logs, n)
        params = SkillParams.calibrate_from_logs(logs, n)
    else:
        n = args.n_skills
        graph = SkillGraph.from_csv(args.graph, n) if args.graph else SkillGraph.random_dag(n, seed=args.seed)
        params = SkillParams.default(n, graph, seed=args.seed)
    bank = ItemBank.synthetic(graph, params.b_center, seed=args.seed)
    return graph, params, bank, RetrievalAgent(bank), logs


def save_json(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {path}")
    return path
