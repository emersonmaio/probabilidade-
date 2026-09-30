import numpy as np
from src.baseline import theoretical_metrics, simulate_random_games, summarize_hits
from src.research import bootstrap_ci, require_temporal_separation

def test_lotomania_theoretical_mean():
    m=theoretical_metrics()
    assert abs(m["mean"]-10.0) < 1e-12
    assert 0 < m["ge16_rate"] < m["ge15_rate"] < 1

def test_random_simulation_shape():
    h=simulate_random_games(n=2000,seed=1)
    s=summarize_hits(h)
    assert s["n"] == 2000
    assert 8.5 < s["mean"] < 11.5
    assert 0 <= s["ge16_rate"] <= 1

def test_bootstrap_ci():
    x=np.arange(100,dtype=float)
    est,lo,hi=bootstrap_ci(x,samples=1000,seed=1)
    assert lo <= est <= hi

def test_temporal_separation():
    require_temporal_separation(1000,2000,2500,3000)
    try:
        require_temporal_separation(2000,1000,2500,3000)
        assert False
    except ValueError:
        pass
