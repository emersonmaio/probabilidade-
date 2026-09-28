from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .data import load_results, incidence
from .backtest import evaluate_weight_model
from .features import score_numbers, select_game
from .search import search, save_result
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default=str(ROOT/"data"/"lotomania.csv")); sub=p.add_subparsers(dest="cmd",required=True)
    v=sub.add_parser("validate"); v.add_argument("--start",type=int,default=300)
    s=sub.add_parser("search"); s.add_argument("--models",type=int,default=10000); s.add_argument("--start",type=int,default=300); s.add_argument("--output",default=str(ROOT/"backtests"/"best_search.json"))
    pr=sub.add_parser("predict"); pr.add_argument("--games",type=int,default=5)
    a=p.parse_args(); df=load_results(a.data); h=incidence(df)
    weights={"freq_10":0.12,"freq_21":0.16,"freq_50":0.27,"freq_100":0.07,"freq_300":0.04,"recency":0.08,"gap":0.18,"transition":0.05,"co21":0.03}
    if a.cmd=="validate": print(json.dumps(evaluate_weight_model(h,weights,a.start).__dict__,indent=2,ensure_ascii=False))
    elif a.cmd=="search":
        r=search(h,a.models,a.start); Path(a.output).parent.mkdir(parents=True,exist_ok=True); save_result(r,a.output); print(json.dumps(r,indent=2,ensure_ascii=False))
    else:
        score=score_numbers(h,weights)
        for i in range(a.games):
            game=select_game(score+np.random.default_rng(5000+i).normal(0,0.03,100)); print(f"Jogo {i+1}: "+" ".join(f"{x:02d}" for x in game))

if __name__=="__main__": main()
