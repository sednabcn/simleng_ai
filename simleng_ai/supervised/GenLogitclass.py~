#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:17:43 2018
These classes are added to statsmodels package

@author: sedna
"""
import numpy as np
import pandas as pd
from statsmodels import base
from statsmodels.discrete.discrete_model import BinaryModel,BinaryResults,BinaryResultsWrapper
from data_manager.generation import Data_Generation
from statsmodels.datasets import utils as du

class GenLogit(BinaryModel):
    __doc__ = """
    Binary choice genlogit model

%(params)s
    %(extra_params)s

    Attributes
    -----------
    endog : array
        A reference to the endogenous response variable
    exog : array
        A reference to the exogenous design.
        % {'params' : base._model_params_doc,
           'extra_params' : base._missing_param_doc}
    """
    def __init__(self,endog,exog,shape=None):

        self.endog=endog
        self.exog=exog
        self.c=float(shape)
        
        data=pd.DataFrame(self.exog).join(self.endog)
        self.data=du.process_pandas(data, endog_idx=0)
        
       
        if self.exog is not None:
            # assume constant
            er = np.linalg.matrix_rank(self.exog)
            self.df_model = float(er - 1)
            self.df_resid = float(self.exog.shape[0] - er)
        else:
            self.df_model = np.nan
            self.df_resid = np.nan
        
    def cdf(self, X):
        """
        The genlogistic cumulative distribution function

        Parameters
        ----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.
        c :  'c' is as a shape parameter c>0 [c=1 becomes to logistic case]
        Returns
        -------
         1/(1 + exp(-x))^c

        Notes
        ------
        In the genlogit model,

        .. math:: \\Lambda\\left(x^{\\prime}\\beta,c\\right)=\\text{Prob}\\left(Y=1|x\\right)=\\frac{1}{\\left(1+e^{-x^{\\prime}\\beta\\right)^{c}}
        """
        
        X = np.asarray(X)
        return 1/(1+np.exp(-X))**self.c

    def pdf(self, X):
        """
        The logistic probability density function

        Parameters
        -----------
        X : array-like
            `X` is the linear predictor of the logit model.  See notes.
        c : 'c' as a shape parameter
        Returns
        -------
        pdf : ndarray
            The value of the Logit probability mass function, PMF, for each
            point of X. ``np.exp(-x)/(1+np.exp(-X))**2``

        Notes
        -----
        In the genlogit model,

        .. math:: \\lambda\\left(x^{\\prime}\\beta,c\\right)=\\frac{c e^{-x^{\\prime}\\beta}}{\\left(1+e^{-x^{\\prime}\\beta}\\right)^{c+1}}
        """
        
        X = np.asarray(X)
        return self.c*np.exp(-X)/(1+np.exp(-X))**(self.c + 1)

    def loglike(self, params):
        """
        Log-likelihood of genlogit model.

        Parameters
        -----------
        params : array-like
            The parameters of the logit model.
            c : 'c' as a shape parameter
        Returns
        -------
        loglike : float
            The log-likelihood function of the model evaluated at `params`.
            See notes.

        Notes
        ------
        .. math:: \\ln L=\\sum_{i}\\ln\\Lambda\\left({\\prime}\\beta,c\\right)

        Where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        return np.sum( (2*y-1)*np.log(Lcdf) +(1-y)*np.log(Lpdf))
    
    def loglikeobs(self, params):
        """
        Log-likelihood of logit model for each observation.
        c : 'c' as a shape parameter
        Parameters
        -----------
        params : array-like
            The parameters of the logit model. 

        Returns
        -------
        loglike : ndarray (nobs,)
            The log likelihood for each observation of the model evaluated
            at `params`. See Notes

        Notes
        ------
        .. math:: \\ln L=\\sum_{i}\\ln\\Lambda\\left(q_{i}x_{i}^{\\prime}\\beta,c\\right)

        for observations :math:`i=1,...,n`

        where :math:`q=2y-1`. This simplification comes from the fact that the
        logistic distribution is symmetric.
        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        return  (2*y-1)*np.log(Lcdf) +(1-y)*np.log(Lpdf)
       

    def score(self, params):
        """
        Logit model score (gradient) vector of the log-likelihood

        Parameters
        ----------
        params: array-like
            The parameters of the model
         c : 'c' as a shape parameter
        Returns
        -------
        score : ndarray, 1-D
            The score vector of the model, i.e. the first derivative of the
            loglikelihood function, evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L}{\\partial\\beta}=\\sum_{i=1}^{n}\\left(y_{i}-\\Lambda_{i}\\right)x_{i}
        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        Lgrad= (1-y)*(1-2.0*Lcdf) + (2.0*y-1.0)*(Lpdf/Lcdf)
        return np.dot(Lgrad,X)

    def jac(self, params):
        """ 
        Logit model Jacobian of the log-likelihood for each observation

        Parameters
        ----------
        params: array-like
            The parameters of the model

        Returns
        -------
        jac : ndarray, (nobs, k_vars)
            The derivative of the loglikelihood for each observation evaluated
            at `params`.

        Notes
        -----
        .. math:: \\frac{\\partial\\ln L_{i}}{\\partial\\beta}=\\left(y_{i}-\\Lambda_{i}\\right)x_{i}

        for observations :math:`i=1,...,n`

        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        Lgrad= (1-y)*(1-2.0*Lcdf) + (2.0*y-1.0)*(Lpdf/Lcdf)
        return Lgrad[:,None] * X

    def hessian(self, params):
        """
        Logit model Hessian matrix of the log-likelihood

        Parameters
        ----------
        params : array-like
            The parameters of the model
            c: 'c' as shape parameter

        Returns
        -------
        hess : ndarray, (k_vars, k_vars)
            The Hessian, second derivative of loglikelihood function,
            evaluated at `params`

        Notes
        -----
        .. math:: \\frac{\\partial^{2}\\ln L}{\\partial\\beta\\partial\\beta^{\\prime}}=-\\sum_{i}\\Lambda_{i}\\left(1-\\Lambda_{i}\\right)x_{i}x_{i}^{\\prime}
        """
        
        X = self.exog
        y = self.endog
        Lcdf=self.cdf(np.dot(X,params))
        Lpdf=self.pdf(np.dot(X,params))
        Lop=Lpdf/Lcdf
        Lgrad_grad=2.0*(1-y)*Lpdf
        Lgrad_grad+=(1-2.0*y)*(1-2.0*Lcdf)*Lop
        Lgrad_grad+=(2.0*y-1.0)*(Lop**2)
        return -np.dot(Lgrad_grad*X.T,X)


    def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        bnryfit = super(GenLogit, self).fit(start_params=start_params,
                method=method, maxiter=maxiter, full_output=full_output,
                disp=disp, callback=callback, **kwargs)

        discretefit = GenLogitResults(self, bnryfit) 
        
        return GenLogit_output(bnryfit,discretefit,self.endog,self.exog,self.c)

    
class GenLogit_output(GenLogit):
         def __init__(self,bnryfit,discretefit,endog,exog,shape):
            
             self.params=bnryfit.params
             self.bse=bnryfit.bse
             self.resid_generalized=discretefit.resid_generalized()
             self.resid_pearson= discretefit.resid_pearson()
             self.resid_dev=discretefit.resid_dev()
             self.predict_=discretefit.predict()
             self.pred_table=discretefit.pred_table()
             self.resid_response=discretefit.resid_response()
             self.fittedvalues=discretefit.fittedvalues()
             super().__init__(endog,exog,shape)
             
         def summary(self):
               pass
         def predict(self, params=None, exog=None, linear=False):
             """
             Predict response variable of a model given exogenous variables.

             Parameters
             ----------
             params : array_like
             Fitted parameters of the model.
             exog : array_like
             1d or 2d array of exogenous values.  If not supplied, the
             whole exog attribute of the model is used.
             linear : bool, optional
             If True, returns the linear predictor dot(exog,params).  Else,
             returns the value of the cdf at the linear predictor.

             Returns
             -------
             array
             Fitted values at exog.
             """
             
             if exog is None:
                 exog = self.exog
             if params is None:
                 params=self.params
                 if not linear:
                    return self.cdf(np.dot(exog, params)) 
                 else:
                     return np.dot(exog, params)


class GenLogitResults(BinaryResults):
    """__doc__ = _discrete_results_docs % {
        "one_line_description" : "A results class for Logit Model",
                   "extra_attr" : ""}
    """
    
    
    def fittedvalues(self):
        """
        Linear predictor XB.
        """
        return np.dot(self.model.exog, self.params[:self.model.exog.shape[1]])

    #@cache_readonly
    def resid_generalized(self):
        """
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Logit model are defined

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`. This is the same as the `resid_response`
        for the Logit model.
        """
        # Generalized residuals
    
        return self.model.endog - self.predict()

    def resid_dev(self):
        """
        Deviance residuals

        Notes
        -----
        Deviance residuals are defined

        .. math:: d_j = \\pm\\left(2\\left[Y_j\\ln\\left(\\frac{Y_j}{M_jp_j}\\right) + (M_j - Y_j\\ln\\left(\\frac{M_j-Y_j}{M_j(1-p_j)} \\right) \\right] \\right)^{1/2}

        where

        :math:`p_j = cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        #These are the deviance residuals
        #model = self.model
        endog = self.model.endog
        #exog = model.exog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        M = 1
        p = self.predict()
        #Y_0 = np.where(exog == 0)
        #Y_M = np.where(exog == M)
        #NOTE: Common covariate patterns are not yet handled
        res = -(1-endog)*np.sqrt(2*M*np.abs(np.log(1-p))) + \
                endog*np.sqrt(2*M*np.abs(np.log(p)))
        return res


    
    def resid_pearson(self):
        """
        Pearson residuals

        Notes
        -----
        Pearson residuals are defined to be

        .. math:: r_j = \\frac{(y - M_jp_j)}{\\sqrt{M_jp_j(1-p_j)}}

        where :math:`p_j=cdf(X\\beta)` and :math:`M_j` is the total number of
        observations sharing the covariate pattern :math:`j`.

        For now :math:`M_j` is always set to 1.
        """
        # Pearson residuals
        #model = self.model
        endog = self.model.endog
        #exog = model.exog
        # M = # of individuals that share a covariate pattern
        # so M[i] = 2 for i = two share a covariate pattern
        # use unique row pattern?
        M = 1
        p = self.predict()
        return (endog - M*p)/np.sqrt(M*p*(1-p))
  

    
    def resid_response(self):
        """
        The response residuals

        Notes
        -----
        Response residuals are defined to be

        .. math:: y - p

        where :math:`p=cdf(X\\beta)`.
        """
        return self.model.endog - self.predict()
