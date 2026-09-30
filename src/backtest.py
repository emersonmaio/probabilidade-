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
    ge15_rate: float
    ge16_rate: float
    ge17_rate: float
    ge18_rate: float
    ge19_rate: float
    ge20_rate: float

    @property
    def tail_tuple(self):
        return (self.ge20_rate, self.ge19_rate, self.ge18_rate,
                self.ge17_rate, self.ge16_rate, self.ge15_rate)

def evaluate_weight_model(history, weights, start=300, end=None, stride=1):
    end = len(history) if end is None else min(end, len(history))
    hits=[]
    for t in range(start, end, stride):
        game=select_game(score_numbers(history[:t], weights))
        hits.append(len(set(game) & set(np.flatnonzero(history[t]))))
    if not hits:
        raise ValueError("Nenhum ponto de validação")
    a=np.asarray(hits,dtype=float)
    return Metrics(
        mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()),
        minimum=int(a.min()), maximum=int(a.max()),
        above_10_rate=float(np.mean(a>10)),
        ge15_rate=float(np.mean(a>=15)), ge16_rate=float(np.mean(a>=16)),
        ge17_rate=float(np.mean(a>=17)), ge18_rate=float(np.mean(a>=18)),
        ge19_rate=float(np.mean(a>=19)), ge20_rate=float(np.mean(a>=20)),
    )

def compare_metrics(candidate, baseline):
    return {
        "mean_delta": candidate.mean-baseline.mean,
        "ge15_delta": candidate.ge15_rate-baseline.ge15_rate,
        "ge16_delta": candidate.ge16_rate-baseline.ge16_rate,
        "ge17_delta": candidate.ge17_rate-baseline.ge17_rate,
        "ge18_delta": candidate.ge18_rate-baseline.ge18_rate,
        "ge19_delta": candidate.ge19_rate-baseline.ge19_rate,
        "ge20_delta": candidate.ge20_rate-baseline.ge20_rate,
    }
