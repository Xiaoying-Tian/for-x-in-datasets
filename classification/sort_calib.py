"""
Script for fitting logistic regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import log_loss 
from sklearn import cross_validation
import matplotlib.pyplot as plt

class PredError:
    def __init__(self, X, y, classifier, seed, linear_model=False, split=0.1):
        self.X_fit, self.X_test, self.y_fit, self.y_test = \
            cross_validation.train_test_split(X, y, test_size=split, random_state=seed)
        self.classifier = classifier
        self.linear_model = linear_model

    def fit(self, rho=0):
        X, y = self.X_fit, self.y_fit
        self.classifier.fit(X, y)
        n = X.shape[0]
        print log_loss(y, self.classifier.predict_proba(X)), "possibly optimistic"

    def conditional(self, nbins=50):
        probs = self.classifier.predict_proba(self.X_test)[:,1].flatten()
        #score = np.log(probs / (1-probs))
        #score = self.classifier.decision_function(self.X_test).flatten() 
        score = probs
        #bins = np.linspace(score.min(), score.max(), nbins)
        bins = np.percentile(score, list(np.linspace(0,100,nbins)))
        digitized = np.digitize(score, bins) - 1 
        bin_means = np.array([self.y_test[digitized == i].mean() for i in range(len(bins))])
        print np.isnan(bin_means).sum()
        cond_y = bin_means[digitized]
        self.calib = cross_entropy(cond_y, probs) - cross_entropy(cond_y, cond_y) 
        self.sort = cross_entropy(cond_y, cond_y)
        print "actual prediction error: ", cross_entropy(self.y_test, probs)
        print "conditional error calculation: ", self.sort, self.calib
        print "sorting + calibration: ", self.sort + self.calib

def cross_entropy(p, q):
    if np.isnan(p).any() or np.isnan(q).any():
        raise ValueError("conditional mean should not be nan")
    idx = (q != 0) & (q != 1)
    return - (p[idx]*np.log(q[idx]) + (1-p[idx])*np.log(1-q[idx])).sum() / p.shape[0]

def main(split=0.1):
    X = pd.read_csv(os.path.join("../datasets", "SpamBase", "DATA.tsv"), sep="\t")
    y = X[X.columns[0]]
    del X[X.columns[0]]

    seed = np.random.randint(0,10000)
    classifier = LogisticRegression()
    pred = PredError(X, y, classifier, seed, linear_model=True)
    pred.fit()
    pred.conditional()
    
    print "Now GradientBoosting Trees"
    classifier = GradientBoostingClassifier()
    pred = PredError(X, y, classifier, seed)
    pred.fit()
    pred.conditional()

    print "Now RFs"
    classifier = RandomForestClassifier(max_depth=10)
    pred = PredError(X, y, classifier, seed)
    pred.fit()
    pred.conditional()

if __name__ == "__main__":
    main()
