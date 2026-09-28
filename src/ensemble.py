from __future__ import annotations
import numpy as np
from .features import score_numbers, select_game

def ensemble_score(history, models):
    total=np.zeros(100); norm=0.0
    for weights, model_weight in models:
        total += float(model_weight)*score_numbers(history,weights); norm += float(model_weight)
    return total/(norm if norm else 1.0)

def ensemble_game(history, models): return select_game(ensemble_score(history,models))
