#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:46:37 2018

@author: sedna
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from biokit.viz import corrplot
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
from statsmodels.multivariate.pca import PCA as smPCA
from sklearn.decomposition import PCA as skPCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from supervised.metrics_classifier_statsmodels import MetricBinaryClassifier as ac
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from supervised import GenLogitclass

class Statsmodels_simula:
    """Simulation with statsmodels."""

    """
    x--X_test
    y--Y_test
    family--pdf to use in the model
    """
    def __init__(self,name,exog,endog,x,y,par,family,method,par_reg,task):
        # to be overridden in subclasses
        self.name=None
        self.model_name=None
        self.model=None
        self.exog= None
        self.endog=None
        self.x=None
        self.y=None
        self.par=None
        self.columns=None
        self.family=None
        self.method=None
        self.par_reg=[]
        self.task=None
        self.linear=False
        self.misclassif=False
        self.y_calibrated_new=None
        self.GenLogit_shape=None

    def fit(self):

        if self.model_name=='sm.OLS':
             print(self.model_name)
             self.model=sm.OLS(self.endog,self.exog).fit()
        elif self.model_name=='sm.GLM':
             print(self.model_name)
             if self.family=="Binomial":
                 family=sm.families.Binomial()
                 self.model=sm.GLM(self.endog,self.exog,family=family).fit()
        elif self.model_name=='smd.Logit':
                print(self.model_name)
                self.model=smd.Logit(self.endog,self.exog).fit()
        elif self.model_name=='GenLogit':
                print(self.model_name)
                self.model=GenLogitclass.GenLogit(self.endog,self.exog,self.GenLogit_shape).fit()
        elif self.model_name=='smd.MNLogit':
                print(self.model_name)
                self.model=smd.MNLogit(self.endog,self.exog).fit()

        elif self.model_name=='smd.Probit':
                print(self.model_name)
                self.model=smd.Probit(self.endog,self.exog).fit()
        else:
            print("Error in loop")
        return self.model

    def fit_regularized(self):

        if  self.method=='l1' and self.model_name=='smd.Logit':

                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                self.model=smd.Logit(self.endog,self.exog).fit_regularized(method='l1',alpha=self.par_reg[0])
                
        if  self.method=='l1' and self.model_name=='GenLogit':

                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                self.model=GenLogitclass.GenLogit(self.endog,self.exog,self.GenLogit_shape).fit_regularized(method='l1',alpha=self.par_reg[0])
                
        if  self.method=='l1' and self.model_name=='smd.Probit':
                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))

        if  self.method=='l1' and self.model_name=='smd.MNLogit':

                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                self.model=smd.MNLogit(self.endog,self.exog).fit_regularized(method='l1',alpha=self.par_reg[0])

        if self.method=='elastic_net' and self.model_name=='sm.OLS':

                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                self.model=sm.OLS(self.endog,self.exog).fit_regularized(method='elastic_net',alpha=self.par_reg[0],L1wt=self.par_reg[1])
                
        if self.method=='elastic_net' and self.model_name=='sm.GLM':

              if self.family=="Binomial":
                  
                 print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                 family=sm.families.Binomial()
                 self.model=sm.GLM(self.endog,self.exog,family=family).fit_regularized(method='elastic_net',alpha=self.par_reg[0],L1wt=self.par_reg[1])
                 
        return self.model

    def summary_models(self):

        self.z_score=self.model.params[1:]/self.model.bse[1:]
        if self.model_name=='GenLogit':
            self.values=self.model.params
            self.zscore=self.z_score
        else:
            self.values=self.model.params.values
            self.zscore=self.z_score.values
        return self.model_name,self.model.summary(),self.values,\
        self.model.resid_pearson,self.model.fittedvalues,self.model.bse,self.zscore
       
        
    def summary_LS_models(self):
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        if self.model_name=='sm.OLS':
            self.prstd, self.iv_l, self.iv_u = wls_prediction_std(self.model)
        if self.model_name=='sm.GLM':
            self.model.ssr=self.model.pearson_chi2
        return self.model.ssr,self.prstd,self.iv_l,self.iv_u

    def calibrate(self):
        self.y_calibrated=self.model.predict(exog=self.exog)
        return self.y_calibrated

    def predict(self):
        if self.model_name=='sm.GLM':
            self.y_estimated = self.model.predict(exog=self.x,linear=self.linear)
        else:
            self.y_estimated=self.model.predict(exog=self.x)
        return self.y_estimated

    def confusion_matrix(self):

        if (self.misclassif==True) :
            self.y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),self.y_calibrated))
            self.y=np.array(self.endog)
        else:
            self.y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),self.y_estimated))
            self.y=np.array(self.y)

        self.y_estimated_=np.array(self.y_estimated_)

        msa=ac.metrics_binary(self.y,self.y_estimated_,self.misclassif)
        if (self.misclassif==True):
            self.msb=msa
        else:
            self.msb=msa[:7]
        return self.msb

    def roc_curve(self):

        fpr,tpr,_=metrics.roc_curve(self.y,self.y_estimated)
        self.fpr=fpr
        self.tpr=tpr
        return self.fpr,self.tpr


class Statsmodels_linear_filter(Statsmodels_simula):
    """ Simulation with the original data ."""

    def __init__(self,names,exog,endog,x,y,par,family,method,par_reg,task,mis_endog):
            self.names=names
            self.exog=exog
            self.model=None
            self.par=par
            self.endog=endog
            self.x=x
            self.y=y
            self.family=family
            self.method=method
            self.par_reg=par_reg #list with two parameters
            self.task=task
            self.mis_endog=mis_endog
            self.misclassif=False
            
            self.GenLogit_shape=par #Carefully
            
            if len(self.mis_endog)>0:
                 self.misclassif=True

    def  statsmodels_linear_supervised(self):
        """Linear Methods to Supervised prediction."""


        to_model_names=[]
        to_params=[]
        to_residuals=OrderedDict()
        confusion_matrix=[]

        to_fitted_values=OrderedDict()
        to_iv_l=OrderedDict()
        to_iv_u=OrderedDict()
        to_z_score=[]
        to_y_calibrated=OrderedDict()
        to_y_estimated=OrderedDict()
        to_fpr=OrderedDict()
        to_tpr=OrderedDict()
        to_fp_index=OrderedDict()
        to_fn_index=OrderedDict()
        to_mis_endog=OrderedDict()
        #names= ['OLS','GLM','Logit','GenLogit','Probit'] #
        smd_models=['Logit','GenLogit','Probit','MNLogit']
        sm_models=['OLS','GLM']
        for name in self.names:
            self.name=name
            if self.name in smd_models and self.name !='GenLogit':
                self.model_name='smd.'+ str(self.name)
            elif self.name=='GenLogit':
                self.model_name=str(self.name)
            elif self.name in sm_models:
                self.model_name='sm.'+ str(self.name)
                self.linear=False
            else:
                print("Error in model_name selection")

            if self.name=='OLS' or 'GLM':
                self.method='elastic_net'
            else:
                self.method ='l1'
            self.columns=self.exog.columns

            if (len(self.par_reg)==0):
                self.model=Statsmodels_simula.fit(self)
            else:
                self.model=Statsmodels_simula.fit_regularized(self)

            model_name,summary,params,resid_pearson,fitted_values,\
            bse,z_score=Statsmodels_simula.summary_models(self)

            if (self.task=="LinearRegression"):
                ssr,prstd,iv_l,iv_u=Statsmodels_simula.summary_LS_models(self)
                to_iv_l[str(self.model_name)]=iv_l
                to_iv_u[str(self.model_name)]=iv_u

            to_model_names.append(model_name)
            y_calibrated=Statsmodels_simula.calibrate(self)
            y_estimated=Statsmodels_simula.predict(self)

            
            #print(y_calibrated.shape)
            #print(y_estimated.shape)
            
            to_y_calibrated[str(self.model_name)]=y_calibrated
            to_y_estimated[str(self.model_name)]=y_estimated
            to_params.append(params)
            to_residuals[str(self.model_name)]=resid_pearson
            to_fitted_values[str(self.model_name)]=fitted_values

            to_z_score.append(z_score)

            #print(input("STOP"))
            
            if (self.task=="BinaryClassification"):

                    sc=Statsmodels_simula.confusion_matrix(self)

                    if self.misclassif==True:
                        confusion_matrix.append(sc[:8])
                        to_fp_index[str(self.model_name)]=sc[10]
                        to_fn_index[str(self.model_name)]=sc[11]
                        to_mis_endog[str(self.model_name)]\
                        =Statsmodels_simula.simple_corrected_mis_classification(self)

                    else:
                        confusion_matrix.append(sc)
                        fpr,tpr=Statsmodels_simula.roc_curve(self)
                        to_fpr[str(self.model_name)]=fpr
                        to_tpr[str(self.model_name)]=tpr



        if (self.task=="BinaryClassification"):

             if (self.misclassif==True):
                 FP_Index=pd.DataFrame.from_dict(to_fp_index,orient='index')
                 FN_Index=pd.DataFrame.from_dict(to_fn_index,orient='index')
                 LEN=[len(names) for names in to_mis_endog.values()]

                 mis_endog_table=pd.DataFrame.from_dict(to_mis_endog,orient='index')

                 mis_endog_table=mis_endog_table.T
                 newindex=range(1,LEN[0]+1)
                 mis_endog_table=pd.DataFrame(np.array(mis_endog_table),index=newindex,\
                                               columns=mis_endog_table.columns)
                 confusion_matrix = pd.DataFrame(confusion_matrix, \
                 index=to_model_names,\
                 #acc,TP,TN,FP,FN,BIAS,pt1,pp1,TNTP,total,fp_index,fn_index
                 columns=['ACC','TP','TN','FP','FN','BIAS','P[X=1]','P[X*=1]'])
             else:

                 to_fpr['AUC=0.5']=fpr
                 to_tpr['AUC=0.5']=fpr
                 FPR=pd.DataFrame.from_dict(to_fpr,orient='index')
                 TPR=pd.DataFrame.from_dict(to_tpr,orient='index')
                 confusion_matrix = pd.DataFrame(confusion_matrix, \
                 index=to_model_names,columns=['ACC','TPR','TNR','PPV','FPR','FNR','DOR'])

        elif (self.task=="LinearRegression"):
            iv_l_table = pd.DataFrame.from_dict(to_iv_l)
            iv_u_table = pd.DataFrame.from_dict(to_iv_u)
        else:
            pass
        to_params=np.array(to_params).reshape(len(to_model_names),len(self.columns)).T
        params_table = pd.DataFrame(to_params,index=self.columns, columns=to_model_names)

        residuals_table = pd.DataFrame.from_dict(to_residuals)



        fitted_values_table = pd.DataFrame.from_dict(to_fitted_values)


        y_calibrated_table = pd.DataFrame.from_dict(to_y_calibrated)

        y_estimated_table = pd.DataFrame.from_dict(to_y_estimated)


        to_z_score=np.array(to_z_score).reshape(len(to_model_names),len(self.columns)-1).T
        z_score_table = pd.DataFrame(to_z_score,index=self.columns[1:],columns=to_model_names)


        if (self.task=="BinaryClassification"):
            if (self.misclassif==True):
                return mis_endog_table,y_calibrated_table,y_estimated_table,FP_Index,FN_Index,confusion_matrix,to_model_names
            else:
                return y_calibrated_table,y_estimated_table,params_table,residuals_table, \
                fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,to_model_names
        else:
            return y_calibrated_table,y_estimated_table,params_table,residuals_table,\
            fitted_values_table,z_score_table, iv_l_table, iv_u_table,to_model_names

