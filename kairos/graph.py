"""Domain knowledge graph and mastery frontier (manuscript §4.1.2)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SkillGraph:
    n_skills: int
    prereq_edges: list[tuple[int, int]]              # (u, v): u is a prerequisite of v
    relation_edges: list[tuple[int, int]] = field(default_factory=list)
    names: list[str] | None = None

    def __post_init__(self):
        self._pre = [[] for _ in range(self.n_skills)]
        for u, v in self.prereq_edges:
            self._pre[v].append(u)
        self._depth = self._compute_depth()

    # -- structure ---------------------------------------------------------
    def prereqs(self, v: int) -> list[int]:
        return self._pre[v]

    def depth(self, v: int) -> int:
        return int(self._depth[v])

    def _compute_depth(self) -> np.ndarray:
        depth = np.zeros(self.n_skills, dtype=int)
        for v in self.topological_order():
            if self._pre[v]:
                depth[v] = 1 + max(depth[u] for u in self._pre[v])
        return depth

    def topological_order(self) -> list[int]:
        indeg = np.zeros(self.n_skills, dtype=int)
        children = [[] for _ in range(self.n_skills)]
        for u, v in self.prereq_edges:
            indeg[v] += 1
            children[u].append(v)
        order, stack = [], [v for v in range(self.n_skills) if indeg[v] == 0]
        while stack:
            u = stack.pop()
            order.append(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        if len(order) != self.n_skills:
            raise ValueError("prerequisite graph contains a cycle")
        return order

    def normalized_adjacency(self) -> np.ndarray:
        """Symmetric-normalised adjacency with self loops, for the GCN (Kipf & Welling)."""
        a = np.eye(self.n_skills)
        for u, v in list(self.prereq_edges) + list(self.relation_edges):
            a[u, v] = a[v, u] = 1.0
        d = a.sum(1)
        dinv = 1.0 / np.sqrt(d)
        return dinv[:, None] * a * dinv[None, :]

    # -- frontier -----------------------------------------------------------
    def frontier(self, mastery: np.ndarray, theta: float) -> list[int]:
        """F_t = {v : all prerequisites of v mastered (>= theta) and m_v < theta}."""
        return [v for v in range(self.n_skills)
                if mastery[v] < theta and all(mastery[u] >= theta for u in self._pre[v])]

    def weakest_prereq(self, v: int, mastery: np.ndarray, theta: float) -> int | None:
        weak = [u for u in self._pre[v] if mastery[u] < theta]
        return min(weak, key=lambda u: mastery[u]) if weak else None

    # -- constructors -------------------------------------------------------
    @staticmethod
    def random_dag(n: int, parent_prob: float = 0.3, window: int = 10,
                   max_parents: int = 2, seed: int = 0) -> "SkillGraph":
        rng = np.random.default_rng(seed)
        edges = []
        for v in range(1, n):
            cands = [u for u in range(max(0, v - window), v) if rng.random() < parent_prob]
            for u in rng.permutation(cands)[:max_parents]:
                edges.append((int(u), v))
        return SkillGraph(n, edges)

    @staticmethod
    def from_csv(path: str, n_skills: int) -> "SkillGraph":
        """CSV with columns `prereq,skill` (0-based indices after the loader's re-mapping)."""
        df = pd.read_csv(path)
        return SkillGraph(n_skills, list(zip(df.prereq.astype(int), df.skill.astype(int))))

    @staticmethod
    def infer_from_logs(df: pd.DataFrame, n_skills: int, min_support: int = 30,
                        order_frac: float = 0.75, max_parents: int = 3) -> "SkillGraph":
        """Heuristic prerequisite inference for datasets without an expert graph (e.g. ASSISTments).

        u -> v if (i) among learners who practised both, >= order_frac practised u first, and
        (ii) accuracy on v is higher for learners whose accuracy on u was above median.
        Edges only go forward in mean first-occurrence order, so the result is a DAG.
        This is a stand-in for an expert-authored graph; the paper must say which was used.
        """
        first = df.groupby(["user", "skill"]).seq_idx.min().unstack()
        acc = df.groupby(["user", "skill"]).correct.mean().unstack()
        mean_first = first.mean().reindex(range(n_skills)).fillna(np.inf).values
        rank = np.argsort(np.argsort(mean_first))
        cols = list(first.columns)
        F, A = first[cols].values, acc[cols].values
        scores: dict[int, list[tuple[float, int]]] = {v: [] for v in range(n_skills)}
        for i, u in enumerate(cols):
            for j, v in enumerate(cols):
                if u == v or rank[u] >= rank[v]:
                    continue
                both = ~np.isnan(F[:, i]) & ~np.isnan(F[:, j])
                if both.sum() < min_support:
                    continue
                if (F[both, i] < F[both, j]).mean() < order_frac:
                    continue
                au, av = A[both, i], A[both, j]
                hi = au >= np.median(au)
                if hi.all() or (~hi).all():
                    continue
                lift = av[hi].mean() - av[~hi].mean()
                if lift > 0.05:
                    scores[v].append((lift, u))
        edges = [(u, v) for v, lst in scores.items() for _, u in sorted(lst, reverse=True)[:max_parents]]
        return SkillGraph(n_skills, edges)
