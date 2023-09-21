#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:37:31 2018

@author: sedna
"""


from sklearn import metrics
import numpy as np


class MetricsMultiClassifier:
    """Confusion matrix in statsmodels"""

    def __init__(self):
        self._paramaters = None

    def metrics_multi_classifier(truth, predicted, misclassif,average,labels=None):
    
        if truth.shape != predicted.shape:
            raise Exception(" Wrong sizes ... ")
        total = truth.shape[0]
        if total == 0:
            return 0

        acc_ = metrics.accuracy_score(truth, predicted)
        acc=np.mean(acc_)
        # The overall precision an recall
        
        ppv_ = metrics.precision_score(truth,predicted,average=average)
        ppv=np.mean(ppv_)
        # 11.12. Metrics of classification performance evaluation

        # TPR [sensitivity]

        tpr_=metrics.recall_score(truth, predicted, average=average)
        tpr=np.mean(tpr_) 
        #FNR [miss rate]

        fnr=1-tpr
        
        # TNR [specificity]

        tnr_=metrics.recall_score(truth, predicted, average=average,pos_label=0)
        tnr=np.mean(tnr_)
        # FPR [fall_out]

        fpr=1 - tnr
        
        # f1_score

        f1_=metrics.f1_score(truth, predicted, average=average)
        f1=np.mean(f1_)
        
        # The overall precision an recall on each individual class[CHECKING]
        p, r, f, s = metrics.precision_recall_fscore_support(truth, predicted, average=average)


        
        #metrics.recall_score(truth, predicted,average,labels)
        # Recalls on individual classes: 

        recalls = metrics.recall_score(truth, predicted, average=None)
       
        # Balanced accuracy

        b_acc = recalls.mean()

                
        ff = 100 * np.array([acc, tpr, tnr, ppv, fpr, fnr, b_acc, f1, np.mean(s)])
                
        acc, tpr, tnr, ppv, fpr, fnr, b_acc, f1, s1 = [float("%.3f" % (a)) for a in ff]

                
        if misclassif == True: # Check for unbalanced dataset
            return acc, tpr, tnr, ppv, fpr, fnr, b_acc, f1, int(s1)
        else:
            return acc, tpr, tnr, ppv, fpr, fnr, b_acc, f1, int(s1)
