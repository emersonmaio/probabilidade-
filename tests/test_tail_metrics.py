import numpy as np
from src.backtest import Metrics, evaluate_weight_model

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
