#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:37:31 2018

@author: sedna
"""


from sklearn import metrics
import numpy as np


class MetricBinaryClassifier:
    """Confusion matrix in statsmodels"""

    def __init__(self):
        self._paramaters = None

    def metrics_binary(truth, predicted, misclassif, treshold):
        if len(truth) != len(predicted):
            raise Exception(" Wrong sizes ... ")
        total = len(truth)
        if total == 0:
            return 0

        acc = metrics.accuracy_score(truth, predicted)
        # The overall precision an recall
        ppv = metrics.precision_score(truth, predicted)
        # 11.12. Metrics of classification performance evaluation

        metrics.recall_score(truth, predicted)
        # Recalls on individual classes: SEN & SPC
        recalls = metrics.recall_score(truth, predicted, average=None)
        tnr = recalls[0]  # is the recall of class 0:
        tpr = recalls[1]  # is the recall of class 1:
        fpr = 1 - tnr
        fnr = 1 - tpr
        # Balanced accuracy
        b_acc = recalls.mean()
        # The overall precision an recall on each individual class[CHECKING]
        p, r, f, s = metrics.precision_recall_fscore_support(truth, predicted)

        ff = 100 * np.array([acc, tpr, tnr, ppv, fpr, fnr, b_acc, f, s])
        acc, tpr, tnr, ppv, fpr, fnr, b_acc, f, s = [float("%.3f" % (a)) for a in ff]

        if misclassif == True:
            return acc, tpr, tnr, ppv, fpr, fnr, b_acc, f, int(s)
        else:
            return acc, tpr, tnr, ppv, fpr, fnr, b_acc, f, int(s)
