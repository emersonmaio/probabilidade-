import numpy as np
from src.features import select_game, score_numbers

def test_select_game():
    g=select_game(np.arange(100,dtype=float)); assert len(g)==50; assert len(set(g))==50; assert all(0<=x<=99 for x in g)

def test_score_shape():
    h=np.zeros((20,100),dtype=np.int8)
    for i in range(20): h[i,(i*3)%100]=1; h[i,(i*3+1)%100]=1
    s=score_numbers(h,{"freq_10":1.0}); assert s.shape==(100,); assert np.isfinite(s).all()
