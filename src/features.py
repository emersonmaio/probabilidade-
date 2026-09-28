from __future__ import annotations
import numpy as np
import pandas as pd
WINDOWS = (10, 21, 50, 100, 300)

def zscore(v):
    v = np.asarray(v, dtype=float)
    s = float(v.std())
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)

def build_features(history: np.ndarray) -> pd.DataFrame:
    n = history.shape[0]
    if n == 0: raise ValueError("Histórico vazio")
    out = {}
    for w in WINDOWS:
        out[f"freq_{w}"] = history[-min(w, n):].mean(axis=0)
    last_seen = np.full(100, n + 1, dtype=int)
    for i in range(n - 1, -1, -1):
        ids = np.flatnonzero(history[i])
        unset = last_seen[ids] == n + 1
        last_seen[ids[unset]] = n - i
    gap = last_seen.astype(float) - 1.0
    out["recency"] = 1.0 / (1.0 + gap)
    out["gap"] = gap / max(1.0, gap.max())
    out["gap_raw"] = gap
    out["repeat_last"] = history[-1].astype(float)
    if n >= 2:
        prev = history[:-1].astype(float)
        cur = history[1:].astype(float)
        trans = cur.T @ prev
        out["transition"] = trans.sum(axis=1) / max(1.0, prev.sum())
    else:
        out["transition"] = np.zeros(100)
    sub = history[-min(21, n):].astype(float)
    target = history[-1].astype(float)
    co = np.zeros(100)
    for a in np.flatnonzero(target): co += sub[:, a] @ sub
    out["co21"] = co / max(1, sub.shape[0])
    out["even"] = np.array([float(x % 2 == 0) for x in range(100)])
    out["decade"] = np.repeat(np.arange(10), 10).astype(float)
    out["momentum"] = out["freq_21"] - out["freq_100"]
    return pd.DataFrame(out)

def score_numbers(history, weights):
    ff = build_features(history)
    score = np.zeros(100)
    for name, weight in weights.items():
        if name in ff: score += float(weight) * zscore(ff[name].to_numpy())
    return score

def select_game(scores, size=50, min_per_decade=4):
    if size != 50: raise ValueError("Este projeto usa jogos de 50 dezenas")
    selected=[]; seen=set()
    for decade in range(10):
        ids=np.arange(decade*10, decade*10+10)
        for x in ids[np.argsort(scores[ids])[::-1]][:min_per_decade]:
            selected.append(int(x)); seen.add(int(x))
    for x in np.argsort(scores)[::-1]:
        x=int(x)
        if len(selected)>=size: break
        if x not in seen: selected.append(x); seen.add(x)
    return sorted(selected[:size])
