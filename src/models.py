from __future__ import annotations
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def fit_logistic(X,y):
    m=LogisticRegression(max_iter=1000,class_weight="balanced"); m.fit(X,y); return m

def fit_extra_trees(X,y,seed=20260924):
    m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=5,class_weight="balanced",random_state=seed,n_jobs=-1); m.fit(X,y); return m

def fit_hist_gradient(X,y,seed=20260924):
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,random_state=seed); m.fit(X,y); return m
