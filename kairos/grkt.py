"""Graph-regularised knowledge tracing (manuscript §4.1.1) and baselines.

GRKT:  h_t = Enc(x_1:t) (causal attentive encoder), Z = GCN(G, X_skill),
       m_{t,v} = sigmoid((<h_t, z_v> + beta_v) / T)
Loss:  L = BCE_next + l1 * L_prereq + l2 * L_smooth + l3 * L_calib
       L_prereq = mean over (t, u->v) of [m_{t,v} - m_{t,u} - xi]_+     (corrected hinge, R2.5)
       L_calib  = soft-binned ECE                                        (R2.5)
Baselines: DKT (LSTM; Piech et al. 2015) and AttnKT (the same attentive encoder without graph
or regularisers; an AKT-style reimplementation -- for the official AKT use pyKT).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .graph import SkillGraph


@dataclass
class GRKTConfig:
    d: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    lambda_prereq: float = 0.3
    lambda_smooth: float = 0.1
    lambda_calib: float = 0.2
    xi: float = 0.05
    calib_bins: int = 10
    max_len: int = 200
    use_graph: bool = True


def pad_batch(seqs, device="cpu"):
    T = max(len(s) for s, _ in seqs)
    S = torch.zeros(len(seqs), T, dtype=torch.long)
    C = torch.zeros(len(seqs), T, dtype=torch.long)
    M = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, (s, c) in enumerate(seqs):
        S[i, :len(s)], C[i, :len(c)], M[i, :len(s)] = torch.as_tensor(s), torch.as_tensor(c), True
    return S.to(device), C.to(device), M.to(device)


def soft_binned_ece(p: torch.Tensor, y: torch.Tensor, bins: int = 10, temp: float = 0.05) -> torch.Tensor:
    centers = (torch.arange(bins, device=p.device, dtype=p.dtype) + 0.5) / bins
    w = torch.softmax(-((p[:, None] - centers[None, :]) ** 2) / (temp ** 2), dim=1)  # N x B
    mass = w.sum(0) + 1e-8
    acc = (w * y[:, None]).sum(0) / mass
    conf = (w * p[:, None]).sum(0) / mass
    return ((mass / p.numel()) * (acc - conf).abs()).sum()


def ece_score(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


class GCN(nn.Module):
    def __init__(self, adj: np.ndarray, d: int):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj, dtype=torch.float32))
        self.X = nn.Parameter(torch.randn(adj.shape[0], d) * 0.1)
        self.W1, self.W2 = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self):
        return self.A @ self.W2(F.relu(self.A @ self.W1(self.X))) + self.X


class GRKT(nn.Module):
    def __init__(self, graph: SkillGraph, cfg: GRKTConfig = GRKTConfig()):
        super().__init__()
        self.graph, self.cfg, V, d = graph, cfg, graph.n_skills, cfg.d
        self.qa_emb = nn.Embedding(2 * V, d)
        self.pos = nn.Embedding(cfg.max_len, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 2 * d, cfg.dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.use_graph:
            self.gcn = GCN(graph.normalized_adjacency(), d)
        else:
            self.Z = nn.Parameter(torch.randn(V, d) * 0.1)
        self.beta = nn.Parameter(torch.zeros(V))
        self.log_T = nn.Parameter(torch.zeros(()), requires_grad=False)   # post-hoc temperature
        pre = graph.prereq_edges or [(0, 0)]
        self.register_buffer("pre_u", torch.tensor([u for u, _ in pre]))
        self.register_buffer("pre_v", torch.tensor([v for _, v in pre]))
        self.has_pre = bool(graph.prereq_edges)

    def skill_embeddings(self):
        return self.gcn() if self.cfg.use_graph else self.Z

    def mastery_logits(self, S, C):
        """Logits of m_{t,v} for every t and v: (B, T, V). Row t uses interactions 1..t."""
        B, T = S.shape
        x = self.qa_emb(S * 2 + C) + self.pos(torch.arange(T, device=S.device))[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=S.device), 1)
        h = self.drop(self.enc(x, mask=causal))
        return (h @ self.skill_embeddings().T + self.beta) / self.log_T.exp()

    def loss(self, S, C, M):
        cfg = self.cfg
        logits = self.mastery_logits(S, C)                       # (B,T,V)
        nxt = logits[:, :-1].gather(2, S[:, 1:, None]).squeeze(2)  # predict r_{t+1}
        valid = M[:, 1:]
        y = C[:, 1:].float()
        bce = F.binary_cross_entropy_with_logits(nxt[valid], y[valid])
        m = torch.sigmoid(logits)
        total, parts = bce, {"bce": bce.item()}
        if self.has_pre and cfg.lambda_prereq > 0:
            hinge = F.relu(m[..., self.pre_v] - m[..., self.pre_u] - cfg.xi)[M]
            total = total + cfg.lambda_prereq * hinge.mean()
            parts["prereq"] = hinge.mean().item()
        if cfg.lambda_smooth > 0:
            sm = ((m[:, 1:] - m[:, :-1]) ** 2).mean(-1)[valid]
            total = total + cfg.lambda_smooth * sm.mean()
            parts["smooth"] = sm.mean().item()
        if cfg.lambda_calib > 0:
            ce = soft_binned_ece(torch.sigmoid(nxt[valid]), y[valid], cfg.calib_bins)
            total = total + cfg.lambda_calib * ce
            parts["calib"] = ce.item()
        return total, parts

    @torch.no_grad()
    def predict_next(self, S, C, M):
        nxt = self.mastery_logits(S, C)[:, :-1].gather(2, S[:, 1:, None]).squeeze(2)
        return torch.sigmoid(nxt), M[:, 1:]


class DKT(nn.Module):
    """Deep knowledge tracing baseline (Piech et al. 2015)."""
    def __init__(self, n_skills: int, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(2 * n_skills, d)
        self.rnn = nn.LSTM(d, d, batch_first=True)
        self.out = nn.Linear(d, n_skills)
        self.drop = nn.Dropout(dropout)
        self.log_T = nn.Parameter(torch.zeros(()), requires_grad=False)

    def mastery_logits(self, S, C):
        h, _ = self.rnn(self.emb(S * 2 + C))
        return self.out(self.drop(h)) / self.log_T.exp()

    def loss(self, S, C, M):
        nxt = self.mastery_logits(S, C)[:, :-1].gather(2, S[:, 1:, None]).squeeze(2)
        valid = M[:, 1:]
        bce = F.binary_cross_entropy_with_logits(nxt[valid], C[:, 1:].float()[valid])
        return bce, {"bce": bce.item()}

    predict_next = GRKT.predict_next


def train_model(model, train_seqs, val_seqs, epochs=10, batch=32, lr=1e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_seqs))
        for i in range(0, len(order), batch):
            S, C, M = pad_batch([train_seqs[j] for j in order[i:i + batch]])
            loss, _ = model.loss(S, C, M)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = evaluate(model, val_seqs)["auc"]
        if verbose:
            print(f"  epoch {ep + 1}: val AUC {auc:.4f}")
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def collect_logits(model, seqs, batch=64):
    model.eval()
    zs, ys = [], []
    for i in range(0, len(seqs), batch):
        S, C, M = pad_batch(seqs[i:i + batch])
        nxt = model.mastery_logits(S, C)[:, :-1].gather(2, S[:, 1:, None]).squeeze(2) * model.log_T.exp()
        zs.append(nxt[M[:, 1:]]); ys.append(C[:, 1:][M[:, 1:]].float())
    return torch.cat(zs), torch.cat(ys)


def fit_temperature(model, val_seqs):
    """Post-hoc temperature scaling (Guo et al. 2017) on validation data."""
    z, y = collect_logits(model, val_seqs)
    log_T = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(z / log_T.exp(), y); l.backward(); return l
    opt.step(closure)
    model.log_T.data = log_T.detach().clamp(-2, 2)
    return float(model.log_T.exp())


@torch.no_grad()
def evaluate(model, seqs, graph: SkillGraph | None = None, xi: float = 0.05, batch=64) -> dict:
    model.eval()
    ps, ys, skip_p, skip_y = [], [], [], []
    viol, n_pairs = 0.0, 0
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        S, C, M = pad_batch(chunk)
        logits = model.mastery_logits(S, C)
        nxt = torch.sigmoid(logits[:, :-1].gather(2, S[:, 1:, None]).squeeze(2))
        valid = M[:, 1:]
        ps.append(nxt[valid].numpy()); ys.append(C[:, 1:][valid].numpy())
        if graph is not None and graph.prereq_edges:
            m = torch.sigmoid(logits)[M]                                    # (N, V)
            u = torch.tensor([a for a, _ in graph.prereq_edges]); v = torch.tensor([b for _, b in graph.prereq_edges])
            viol += float((m[:, v] > m[:, u] + xi).float().sum()); n_pairs += m.shape[0] * len(u)
            # prerequisite-skipping subset: attempts on v made before any attempt on a prereq of v
            for bi, (s, c) in enumerate(chunk):
                seen = set()
                for t in range(1, len(s)):
                    seen.add(int(s[t - 1]))
                    pre = graph.prereqs(int(s[t]))
                    if pre and not any(p in seen for p in pre):
                        skip_p.append(float(nxt[bi, t - 1])); skip_y.append(int(c[t]))
    p, y = np.concatenate(ps), np.concatenate(ys)
    out = {"auc": float(roc_auc_score(y, p)), "ece": ece_score(p, y), "n": int(len(y))}
    if n_pairs:
        out["prereq_consistency"] = 1 - viol / n_pairs
    if len(set(skip_y)) == 2:
        out["auc_prereq_skip"] = float(roc_auc_score(skip_y, skip_p)); out["n_prereq_skip"] = len(skip_y)
    return out
