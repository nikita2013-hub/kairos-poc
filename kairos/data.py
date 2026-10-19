"""Loaders for public response logs (manuscript Table 7)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def load_assistments_2009(path: str, max_skills: int = 100, min_len: int = 10) -> tuple[pd.DataFrame, dict]:
    """ASSISTments 2009-2010 skill-builder CSV (columns: user_id, order_id, skill_id, correct,
    ms_first_response, hint_count). Keeps the `max_skills` most frequent skills.
    Returns a frame with columns user, skill (0..V-1), correct, seq_idx, latency_ms, hints."""
    raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
    raw = raw.dropna(subset=["skill_id"])
    raw = raw[raw.original == 1] if "original" in raw.columns else raw
    top = raw.skill_id.value_counts().index[:max_skills]
    raw = raw[raw.skill_id.isin(top)].sort_values(["user_id", "order_id"])
    skill_map = {s: i for i, s in enumerate(sorted(top))}
    df = pd.DataFrame({
        "user": raw.user_id.values,
        "skill": raw.skill_id.map(skill_map).values.astype(int),
        "correct": raw.correct.clip(0, 1).astype(int).values,
        "latency_ms": raw.get("ms_first_response", pd.Series(np.nan, index=raw.index)).values,
        "hints": raw.get("hint_count", pd.Series(0, index=raw.index)).values,
    })
    df["seq_idx"] = df.groupby("user").cumcount()
    lens = df.groupby("user").size()
    df = df[df.user.isin(lens[lens >= min_len].index)].reset_index(drop=True)
    return df, skill_map


def load_ednet_kt1(folder: str, question_csv: str, max_users: int = 5000,
                   max_skills: int = 100, min_len: int = 10) -> tuple[pd.DataFrame, dict]:
    """EdNet-KT1: one CSV per user (timestamp, question_id, user_answer, elapsed_time) plus
    contents/questions.csv (question_id, correct_answer, tags). First tag is used as skill."""
    import glob, os
    q = pd.read_csv(question_csv)
    q["skill_raw"] = q.tags.astype(str).str.split(";").str[0]
    files = sorted(glob.glob(os.path.join(folder, "u*.csv")))[:max_users]
    frames = []
    for f in files:
        u = pd.read_csv(f).merge(q[["question_id", "correct_answer", "skill_raw"]], on="question_id")
        u["user"] = os.path.basename(f)[:-4]
        u["correct"] = (u.user_answer == u.correct_answer).astype(int)
        u["latency_ms"] = u.elapsed_time
        frames.append(u.sort_values("timestamp"))
    raw = pd.concat(frames)
    top = raw.skill_raw.value_counts().index[:max_skills]
    raw = raw[raw.skill_raw.isin(top)]
    skill_map = {s: i for i, s in enumerate(sorted(top))}
    df = pd.DataFrame({"user": raw.user.values, "skill": raw.skill_raw.map(skill_map).values,
                       "correct": raw.correct.values, "latency_ms": raw.latency_ms.values, "hints": 0})
    df["seq_idx"] = df.groupby("user").cumcount()
    lens = df.groupby("user").size()
    return df[df.user.isin(lens[lens >= min_len].index)].reset_index(drop=True), skill_map


def to_sequences(df: pd.DataFrame, max_len: int = 200) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split each learner's log into chunks of at most max_len interactions."""
    seqs = []
    for _, g in df.sort_values(["user", "seq_idx"]).groupby("user"):
        s, c = g.skill.values.astype(np.int64), g.correct.values.astype(np.int64)
        for i in range(0, len(s), max_len):
            if len(s[i:i + max_len]) >= 3:
                seqs.append((s[i:i + max_len], c[i:i + max_len]))
    return seqs


def split_by_learner(df: pd.DataFrame, frac=(0.7, 0.1, 0.2), seed: int = 0):
    users = df.user.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n1, n2 = int(frac[0] * len(users)), int((frac[0] + frac[1]) * len(users))
    parts = users[:n1], users[n1:n2], users[n2:]
    return [df[df.user.isin(p)].reset_index(drop=True) for p in parts]
