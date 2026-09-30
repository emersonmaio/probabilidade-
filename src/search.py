from __future__ import annotations
import json
import numpy as np
from .backtest import evaluate_weight_model, evaluate_weight_models_cached

FEATURES=["freq_10","freq_21","freq_50","freq_100","freq_300",
          "recency","gap","transition","co21","repeat_last","momentum"]

def random_weights(rng):
    w=rng.dirichlet(np.ones(len(FEATURES)))
    return dict(zip(FEATURES,map(float,w)))

def objective(m):
    # Tail-first objective; mean is a secondary tie-breaker.
    return (m.ge20_rate, m.ge19_rate, m.ge18_rate, m.ge17_rate,
            m.ge16_rate, m.ge15_rate, m.mean, -m.std)

def search(history, models=10000, start=300, end=None, seed=20260924):
    rng=np.random.default_rng(seed)
    candidates=[random_weights(rng) for _ in range(models)]
    metrics=evaluate_weight_models_cached(history,candidates,FEATURES,start=start,end=end)
    best=None
    for i,(weights,m) in enumerate(zip(candidates,metrics),start=1):
        row={"model_id":i,"objective":objective(m),
             "metrics":m.__dict__,"weights":weights}
        if best is None or row["objective"]>best["objective"]:
            best=row
    return best

def nested_search(history, train_start=300, train_end=None, validation_end=None,
                  models=10000, seed=20260924):
    n=len(history)
    train_end=n if train_end is None else min(train_end,n)
    validation_end=n if validation_end is None else min(validation_end,n)
    if train_end <= train_start or validation_end <= train_end:
        raise ValueError("Blocos temporais inválidos")
    best=search(history[:train_end],models=models,start=train_start,seed=seed)
    validation=evaluate_weight_model(history,best["weights"],
                                     start=train_end,end=validation_end)
    best["validation_metrics"]=validation.__dict__
    return best

def save_result(result,path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
