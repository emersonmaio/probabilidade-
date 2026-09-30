import numpy as np
from src.backtest import Metrics, evaluate_weight_model, evaluate_weight_models_cached
from src.search import FEATURES

def test_tail_metrics():
    m=Metrics(10,10,2,3,19,0.5,0.10,0.05,0.02,0.01,0.005,0.001)
    assert m.tail_tuple == (0.001,0.005,0.01,0.02,0.05,0.10)

def test_evaluate_weight_model_tail_fields():
    h=np.zeros((305,100),dtype=np.int8)
    for i in range(305):
        h[i, i % 100]=1
    m=evaluate_weight_model(h,{"freq_10":1.0},start=300)
    assert 0 <= m.ge15_rate <= 1
    assert 0 <= m.ge20_rate <= 1
    assert m.maximum <= 20

def test_cached_matches_single_model():
    rng=np.random.default_rng(7)
    h=np.zeros((340,100),dtype=np.int8)
    for i in range(340):
        h[i,rng.choice(100,size=20,replace=False)]=1
    weights={"freq_10":0.4,"freq_50":0.6}
    single=evaluate_weight_model(h,weights,start=300,end=330)
    cached=evaluate_weight_models_cached(h,[weights],FEATURES,start=300,end=330)[0]
    assert np.isclose(single.mean,cached.mean)
    assert single.maximum==cached.maximum
    assert np.isclose(single.ge16_rate,cached.ge16_rate)
