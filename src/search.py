from __future__ import annotations
import json
import numpy as np
from .backtest import evaluate_weight_model
FEATURES=["freq_10","freq_21","freq_50","freq_100","freq_300","recency","gap","transition","co21","repeat_last","momentum"]

def random_weights(rng):
    w=rng.dirichlet(np.ones(len(FEATURES)))
    return dict(zip(FEATURES,map(float,w)))

def objective(m):
    return m.mean - 0.20*m.std + 0.50*(m.above_10_rate-0.50)

def search(history, models=10000, start=300, seed=20260924):
    rng=np.random.default_rng(seed); best=None
    for i in range(models):
        weights=random_weights(rng); m=evaluate_weight_model(history,weights,start=start)
        row={"model_id":i+1,"objective":objective(m),"metrics":m.__dict__,"weights":weights}
        if best is None or row["objective"]>best["objective"]: best=row
    return best

def save_result(result,path):
    with open(path,"w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
