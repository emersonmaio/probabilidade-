from __future__ import annotations
import numpy as np
from .features import score_numbers, select_game

def build_portfolio(history, models, games=5, max_overlap=42):
    if not models: raise ValueError("Nenhum modelo")
    scores=[score_numbers(history,m) for m in models]; output=[]
    for i in range(games):
        base=scores[i%len(scores)].copy()
        for old in output: base[np.asarray(old,dtype=int)] -= 0.20
        candidate=select_game(base)
        if any(len(set(candidate)&set(old))>max_overlap for old in output):
            candidate=select_game(base+np.random.default_rng(1000+i).normal(0,0.05,100))
        output.append(candidate)
    return output
