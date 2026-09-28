from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .features import score_numbers, select_game

@dataclass
class Metrics:
    mean: float
    median: float
    std: float
    minimum: int
    maximum: int
    above_10_rate: float

def evaluate_weight_model(history, weights, start=300, end=None, stride=1):
    end = len(history) if end is None else min(end, len(history))
    hits=[]
    for t in range(start, end, stride):
        game=select_game(score_numbers(history[:t], weights))
        hits.append(len(set(game) & set(np.flatnonzero(history[t]))))
    if not hits: raise ValueError("Nenhum ponto de validação")
    a=np.asarray(hits,dtype=float)
    return Metrics(float(a.mean()),float(np.median(a)),float(a.std()),int(a.min()),int(a.max()),float(np.mean(a>10)))
