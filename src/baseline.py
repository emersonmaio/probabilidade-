from __future__ import annotations
from math import comb
import numpy as np

N=100
DRAW_SIZE=20
GAME_SIZE=50

def theoretical_pmf():
    den=comb(N,GAME_SIZE)
    return {k: comb(DRAW_SIZE,k)*comb(N-DRAW_SIZE,GAME_SIZE-k)/den
            for k in range(0,DRAW_SIZE+1)
            if 0 <= GAME_SIZE-k <= N-DRAW_SIZE}

def theoretical_metrics():
    pmf=theoretical_pmf()
    mean=sum(k*p for k,p in pmf.items())
    return {
        "mean": mean,
        "ge15_rate": sum(p for k,p in pmf.items() if k>=15),
        "ge16_rate": sum(p for k,p in pmf.items() if k>=16),
        "ge17_rate": sum(p for k,p in pmf.items() if k>=17),
        "ge18_rate": sum(p for k,p in pmf.items() if k>=18),
        "ge19_rate": sum(p for k,p in pmf.items() if k>=19),
        "ge20_rate": sum(p for k,p in pmf.items() if k>=20),
        "pmf": pmf,
    }

def simulate_random_games(n=100_000, seed=20260930):
    rng=np.random.default_rng(seed)
    hits=np.empty(n,dtype=np.int16)
    universe=np.arange(N)
    for i in range(n):
        game=rng.choice(universe,size=GAME_SIZE,replace=False)
        draw=rng.choice(universe,size=DRAW_SIZE,replace=False)
        hits[i]=np.intersect1d(game,draw,assume_unique=True).size
    return hits

def summarize_hits(hits):
    a=np.asarray(hits,dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "minimum": int(a.min()),
        "maximum": int(a.max()),
        "ge15_rate": float(np.mean(a>=15)),
        "ge16_rate": float(np.mean(a>=16)),
        "ge17_rate": float(np.mean(a>=17)),
        "ge18_rate": float(np.mean(a>=18)),
        "ge19_rate": float(np.mean(a>=19)),
        "ge20_rate": float(np.mean(a>=20)),
    }
