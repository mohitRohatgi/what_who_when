#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:50:33 2017

@author: m0r00ds
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class Trainer:
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y
    
    def predict(self, test):
        clf, scores = self._train()
        proba = clf.predict_proba(test)
        return clf.predict(test), scores
    
    def _train(self):
#        clf = RandomForestClassifier(n_estimators=5, max_depth=10,
#                                           min_samples_split=10, random_state=0)
        clf = LogisticRegression(C=10, max_iter=10000000)
        clf.fit(self.X, self.y)
        scores = cross_val_score(clf, self.X, self.y)
        return clf, scores