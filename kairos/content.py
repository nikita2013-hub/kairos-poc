"""Item bank and Retrieval Agent (manuscript §4.4).

The PoC uses a templated arithmetic/proportional-reasoning domain so that every item has
machine-checkable canonical claims K(y_ped). Items are keyed to graph nodes; retrieval is
planned by the Pedagogy Agent (target skill + difficulty), never by free-text similarity.
"""
from __future__ import annotations
import numpy as np

from .types import Item
from .graph import SkillGraph

FAMILIES = [
    ("fraction_of", "Find {a}/{b} of {n}.", "compute {a}/{b} of {n}"),
    ("percent_of", "What is {p}% of {n}?", "compute {p} percent of {n}"),
    ("unit_rate", "{n} items cost {c} rupees. What is the cost of one item?", "compute {c} divided by {n}"),
    ("ratio_split", "Share {n} in the ratio {a}:{b}. How large is the first part?", "compute first part of {n} in ratio {a}:{b}"),
    ("area_rect", "A rectangle is {a} m by {n} m. What is its area?", "compute {a} times {n}"),
    ("simple_interest", "Find the simple interest on {n} rupees at {p}% per year for {a} years.",
     "compute simple interest on {n} at {p} percent for {a} years"),
]


def _numbers(rng, level: float):
    """Harder items use larger, less friendly numbers."""
    scale = 1 + int(4 * level)
    a = int(rng.integers(1, 3 + scale))
    b = int(rng.integers(a + 1, a + 4 + scale))
    n = int(b * rng.integers(2, 4 + 3 * scale))
    p = int(rng.choice([10, 20, 25, 50] if level < 0.5 else [12, 15, 35, 45, 65]))
    c = int(n * rng.integers(2, 6 + 5 * scale))
    return dict(a=a, b=b, n=n, p=p, c=c)


class ItemBank:
    def __init__(self, items: list[Item], n_skills: int):
        self.items = items
        self.by_skill: list[list[Item]] = [[] for _ in range(n_skills)]
        for it in items:
            self.by_skill[it.skill].append(it)
        self.d = {it.item_id: 1 / (1 + np.exp(-it.b)) for it in items}

    @staticmethod
    def synthetic(graph: SkillGraph, b_center: np.ndarray, per_skill: int = 30, seed: int = 0) -> "ItemBank":
        rng = np.random.default_rng(seed)
        items = []
        for v in range(graph.n_skills):
            fam, stem_t, claim_t = FAMILIES[v % len(FAMILIES)]
            for k in range(per_skill):
                b = float(b_center[v] + np.linspace(-2.2, 2.2, per_skill)[k] + rng.normal(0, 0.1))
                nums = _numbers(rng, 1 / (1 + np.exp(-b)))
                items.append(Item(f"s{v}_i{k}", v, b, stem_t.format(**nums), (claim_t.format(**nums),)))
        return ItemBank(items, graph.n_skills)

    def difficulty(self, item: Item) -> float:
        return self.d[item.item_id]


class RetrievalAgent:
    """Returns the item for skill v whose difficulty is closest to the requested target,
    excluding recently used items; provenance = (item_id, skill node)."""
    def __init__(self, bank: ItemBank):
        self.bank = bank

    def fetch(self, skill: int, target_d: float, exclude: set[str] = frozenset()) -> Item:
        cands = [it for it in self.bank.by_skill[skill] if it.item_id not in exclude] or self.bank.by_skill[skill]
        return min(cands, key=lambda it: abs(self.bank.difficulty(it) - target_d))
