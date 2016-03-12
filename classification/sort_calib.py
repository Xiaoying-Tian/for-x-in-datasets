"""
Script for fitting logistic regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss 
from sklearn import cross_validation
import matplotlib.pyplot as plt

class PredError:
    def __init__(self, X, y, split=0.1):
        self.X_fit, self.X_test, self.y_fit, self.y_test = \
            cross_validation.train_test_split(X, y, test_size=split)

    def fit(self, rho=0):
        X, y = self.X_fit, self.y_fit
        self.lr = LogisticRegression(C=1)
        self.lr.fit(X, y)
        n = X.shape[0]
        ones = np.where(y==1)[0]
        zeros = np.where(y==0)[0]
        FDR = sum((self.lr.predict(X)==1) & (y==0)) * 1.0 / n 
        FNR = sum((self.lr.predict(X)==0) & (y==1)) * 1.0 / n
        print "FDR:", FDR, " FNR:", FNR
        print log_loss(y, self.lr.predict_proba(X)), "possibly optimistic"

    def conditional(self, nbins=100):
        score = np.dot(self.X_test, self.lr.coef_.flatten())
        bins = np.linspace(score.min(), score.max(), nbins)
        digitized = np.digitize(score, bins) - 1 
        bin_means = np.array([self.y_test[digitized == i].mean() for i in range(len(bins))])
        print np.isnan(bin_means).sum()
        cond_y = bin_means[digitized]
        probs = self.lr.predict_proba(self.X_test)[:,1].flatten()
        self.sort = cross_entropy(cond_y, probs) - cross_entropy(cond_y, cond_y) 
        self.calib = cross_entropy(cond_y, cond_y)
        print "actual prediction error: ", log_loss(self.y_test, self.lr.predict_proba(self.X_test)) 
        print "conditional error calculation: ", self.sort, self.calib

def cross_entropy(p, q):
    if np.isnan(p).any() or np.isnan(q).any():
        raise ValueError("conditional mean should not be nan")
    idx = (p != 0) & (p != 1) & (q != 0) & (q != 1)
    return - (p[idx]*np.log(q[idx]) + (1-p[idx])*np.log(1-q[idx])).sum() / p.shape[0]

def main(split=0.1):
    X = pd.read_csv(os.path.join("../datasets", "SpamBase", "DATA.tsv"), sep="\t")
    y = X[X.columns[0]]
    del X[X.columns[0]]
    pred = PredError(X, y)
    pred.fit()
    pred.conditional()
    

if __name__ == "__main__":
    main()
