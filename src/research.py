from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .backtest import evaluate_weight_model
from .baseline import summarize_hits, simulate_random_games, theoretical_metrics

def bootstrap_ci(values, statistic=np.mean, samples=5000, seed=20260930, alpha=0.05):
    a=np.asarray(values,dtype=float)
    if a.size == 0:
        raise ValueError("Amostra vazia")
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,a.size,size=(samples,a.size))
    stats=statistic(a[idx],axis=1)
    lo,hi=np.quantile(stats,[alpha/2,1-alpha/2])
    return float(statistic(a)),float(lo),float(hi)

def stability_windows(history, weights, windows=(100,300,500,1000), min_start=300):
    out={}
    n=len(history)
    for w in windows:
        if n < w:
            continue
        start=max(min_start,n-w)
        m=evaluate_weight_model(history,weights,start=start,end=n)
        out[str(w)]=asdict(m)
    return out

def walk_forward(history, weights, start=300, end=None, stride=1):
    end=len(history) if end is None else min(end,len(history))
    rows=[]
    for t in range(start,end,stride):
        game=evaluate_weight_model(history,weights,start=t,end=t+1)
        rows.append({"target_index":t,**asdict(game)})
    return rows

def feature_ablation(history, feature_weights, start=300, end=None):
    results=[]
    for name in feature_weights:
        m=evaluate_weight_model(history,{name:1.0},start=start,end=end)
        results.append({"feature":name,**asdict(m)})
    return sorted(results,key=lambda x:(x["ge16_rate"],x["ge15_rate"],x["mean"]),reverse=True)

def grouped_ablation(history, groups, start=300, end=None):
    results=[]
    for name,features in groups.items():
        weights={f:1.0 for f in features}
        m=evaluate_weight_model(history,weights,start=start,end=end)
        results.append({"group":name,"features":list(features),**asdict(m)})
    return sorted(results,key=lambda x:(x["ge16_rate"],x["ge15_rate"],x["mean"]),reverse=True)

def model_report(history, weights, start=300, end=None, bootstrap_samples=5000):
    m=evaluate_weight_model(history,weights,start=start,end=end)
    # Recreate the hit series for bootstrap without changing the production API.
    rows=walk_forward(history,weights,start=start,end=end)
    hits=np.array([r["mean"] for r in rows],dtype=float)
    mean,lo,hi=bootstrap_ci(hits,samples=bootstrap_samples)
    baseline=theoretical_metrics()
    random_hits=simulate_random_games(n=min(100_000,max(10_000,len(rows)*100)),seed=20260930)
    return {
        "metrics":asdict(m),
        "mean_bootstrap_ci":{"estimate":mean,"low":lo,"high":hi},
        "theoretical_baseline":baseline,
        "simulated_baseline":summarize_hits(random_hits),
        "stability":stability_windows(history,weights),
    }

def compare_to_baseline(metrics, baseline=None):
    b=theoretical_metrics() if baseline is None else baseline
    return {
        "mean_delta":metrics.mean-b["mean"],
        "ge15_delta":metrics.ge15_rate-b["ge15_rate"],
        "ge16_delta":metrics.ge16_rate-b["ge16_rate"],
        "ge17_delta":metrics.ge17_rate-b["ge17_rate"],
        "ge18_delta":metrics.ge18_rate-b["ge18_rate"],
        "ge19_delta":metrics.ge19_rate-b["ge19_rate"],
        "ge20_delta":metrics.ge20_rate-b["ge20_rate"],
    }

def require_temporal_separation(train_end, validation_end, holdout_start, n):
    if not (0 < train_end < validation_end <= holdout_start <= n):
        raise ValueError("Blocos temporais devem ser estritamente ordenados")

def nested_temporal_evaluation(history, candidate_weights, train_end, validation_end, holdout_start):
    n=len(history)
    require_temporal_separation(train_end,validation_end,holdout_start,n)
    # Candidate weights are supplied from training research. This function never
    # uses validation/holdout results to alter them.
    validation=evaluate_weight_model(history,candidate_weights,start=train_end,end=validation_end)
    holdout=evaluate_weight_model(history,candidate_weights,start=holdout_start,end=n)
    return {"validation":asdict(validation),"holdout":asdict(holdout)}

def leave_one_feature_out(history, weights, start=300, end=None):
    names=list(weights)
    results=[]
    for name in names:
        reduced={k:v for k,v in weights.items() if k!=name}
        m=evaluate_weight_model(history,reduced,start=start,end=end)
        results.append({"removed":name,**asdict(m)})
    return sorted(results,key=lambda x:(x["ge16_rate"],x["ge15_rate"],x["mean"]),reverse=True)
