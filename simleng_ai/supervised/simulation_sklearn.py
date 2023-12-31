#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
import time

# import nincrease
import seaborn as sns

from functools import wraps
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from biokit.viz import corrplot

from scipy import stats, linalg
from scipy.linalg import svd as sksvd
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# from resources import Tools

from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing
from sklearn import manifold
from sklearn import semi_supervised


from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KernelCenterer,
    LabelBinarizer,
    LabelEncoder,
    MultiLabelBinarizer,
    MinMaxScaler,
    MaxAbsScaler,
    QuantileTransformer,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    add_dummy_feature,
    PolynomialFeatures,
    binarize,
    normalize,
    scale,
    robust_scale,
    maxabs_scale,
    minmax_scale,
    label_binarize,
    quantile_transform,
)


# from nincrease import fit_on_increasing_size,plot_r2_snr
from sklearn.manifold import (
    MDS as skMDS,
    smacof as sksmacof,
    Isomap as skIsomap,
    TSNE as skTSNE,
    spectral_embedding as skSpectEmbed,
)

# LocallyLinearEmbeddig as skLocalLEmbed,spectral_embedding as skSpectEmbed

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA as skPCA
from sklearn.model_selection import GridSearchCV as skGridSCV
from sklearn.model_selection import KFold
from sklearn.model_selection import (
    learning_curve as sklearncur,
    validation_curve as skvalcur,
)


from sklearn.feature_selection import SelectKBest as skSB
from sklearn.feature_selection import f_classif as skf_c
from sklearn.feature_selection import f_regression as skf_r
from sklearn.feature_selection import VarianceThreshold as skVT
from sklearn.feature_selection import RFE as skRFE
from sklearn.feature_selection import RFECV as skRFECV
from sklearn.feature_selection import mutual_info_classif as skmiC
from sklearn.feature_selection import mutual_info_regression as skmiR
from sklearn.feature_selection import SelectFromModel as skFM

# Nov24/2023
#It must include text and images to expand one
from sklearn.feature_extraction import DictVectorizer


from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline


from sklearn import linear_model as sklm
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV
from sklearn.linear_model import (
    Hinge,
    Huber,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
)
from sklearn.linear_model import (
    LinearRegression,
    Log,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.linear_model import (
    ModifiedHuber,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
)
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
)
from sklearn.linear_model import (
    RANSACRegressor,
    Ridge,
    RidgeCV,
    RidgeClassifier,
    RidgeClassifierCV,
)
from sklearn.linear_model import (
    SGDClassifier,
    SGDRegressor,
    SquaredLoss,
    TheilSenRegressor,
)

# RandomizedLasso,RandomizedLogisticRegression
from sklearn.linear_model import (
    enet_path,
    lars_path,
    lasso_path,
    orthogonal_mp,
    orthogonal_mp_gram,
    ridge_regression,
)

# base,bayes, cd_fast, coordinate descent,huber,lasso_stability_path,least_angle,logistic_regression_path,logistic,omp,passive_aggressive,perceptron,randomized_l1,ransac,ridge,sag,sag_fast, sgd_fast,stochastic_gradient, theil_sen
from sklearn.random_projection import (
    BaseRandomProjection,
    GaussianRandomProjection,
    SparseRandomProjection,
)
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from sklearn import svm
from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, LinearSVR
from sklearn.svm import SVC as skSVC
from sklearn.svm import SVR as skSVR
from sklearn.isotonic import IsotonicRegression as skIR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    export_graphviz,
)

from sklearn.gaussian_process import GaussianProcessClassifier as skGPC
from sklearn.gaussian_process import GaussianProcessRegressor as skGPR

# from sklearn.gaussian_process import GaussianProcess as skGP
# from sklearn.gaussian_process.correlation_models import absolute_exponential,squared_exponential,generalized_exponential,pure_nugget,cubic

from sklearn.gaussian_process.kernels import RBF as skRBF, Matern as skMatern
from sklearn.gaussian_process.kernels import (
    RationalQuadratic as skRQ,
    ExpSineSquared as skESS,
    DotProduct as skDP,
    PairwiseKernel as skPK,
)

from sklearn.kernel_approximation import (
    RBFSampler as skRBFSampler,
    SkewedChi2Sampler as skSChi2Sampler,
    AdditiveChi2Sampler as skAdChi2Sampler,
    Nystroem as skNystroem,
)
from sklearn.kernel_ridge import KernelRidge as skKRidge

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as skQDA


from sklearn.mixture import GaussianMixture

# ,Bayesian_Mixture,DPGMM,VBGMM
from sklearn.multiclass import (
    OneVsRestClassifier,
    OneVsOneClassifier,
    OutputCodeClassifier,
)
from sklearn.multioutput import (
    MultiOutputRegressor,
    MultiOutputClassifier,
    ClassifierChain,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.neighbors import (
    BallTree,
    NearestCentroid,
    NearestNeighbors,
    kneighbors_graph,
    radius_neighbors_graph,
    KernelDensity,
    LocalOutlierFactor,
    KDTree,
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsRegressor,
)

# ,DistanceMetric,LSHForest
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor

from sklearn.ensemble import BaggingClassifier as skBagC
from sklearn.ensemble import BaggingRegressor as skBagR
from sklearn.ensemble import RandomForestClassifier as skRFC
from sklearn.ensemble import RandomForestRegressor as skRFR
from sklearn.ensemble import GradientBoostingClassifier as skGBC
from sklearn.ensemble import GradientBoostingRegressor as skGBR
from sklearn.ensemble import IsolationForest as skIF
from sklearn.ensemble import VotingClassifier as skVC
from sklearn.ensemble import AdaBoostClassifier as skABC
from sklearn.ensemble import AdaBoostRegressor as skABR

from sklearn.metrics import balanced_accuracy_score as balanced_acc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from ..supervised.metrics_classifier_sklearn import MetricsMultiClassifier as ac
from ..resources.ml import emb_nclass, inv_emb_nclass

colors = [ic for ic in mcolors.BASE_COLORS.values() if ic != (1, 1, 1)]

class Sklearn_simula:
    """Simulation with statsmodels."""

    """
    x--X_test
    y--Y_test
    family--pdf to use in the model
    """

    def __init__(self,model,exog, endog, x, y):
        # to be overridden in subclasses
        self.name=None
        self.model_names=[]
        self.model_name=None
        self.model_list =[]
        self.model = model
        self.exog = None
        self.endog = None
        self.x = None
        self.y = None
        self.par = None
        self.ntarget=None
        self.nclass=None
        self.missclassif=False
        self.par_reg=[] 
        self.columns = None
           
        
    def fit(self):
        
        return self.model.fit(self.exog,self.endog)

    def partial_fit(self):

        return self.model.partial_fit(self.exog,self.endog)

    def fit_transform(self):
        self.exog_fit_transf=self.model.fit_transform(self.exog)
        return self.exog_fit_transf

    def transform(self):
        self.exog_transf=self.model.fit_transform(self.exog)
        return self.exog_transf
    
        
    def fit_regularized(self):

        return self.model.fit_regularized(self.exog,self.endog)

    def decision_function(self):
        self.decision_on_class =self.model.decision_function(self.exog)
        return self.decision_on_class
    
    def calibrate(self):
        self.y_calibrated = self.model.predict(self.exog)
        return self.y_calibrated

    def predict(self):
        self.y_estimated = self.model.predict(self.x)
        return self.y_estimated

    def predict_proba(self):
        self.y_estimated_proba = self.model.predict_proba(self.x)
        return self.y_estimated_proba

    
    def score(self):
        self.score=self.model.score(self.x,self.y)
        return self.score
    # check 
    def get_params(self):
        self.get_parameters=[parameter for parameter in self.model.get_params()]
        return self.get_parameters
    # check to use to apply functions
    def set_params(self):
        self.set_parameters=self.model.set_params(self.par)
        return self.set_parameters
       
    def confusion_matrix(self):

        if self.misclassif == True and self.nclass==2:
            
            y_endog_ = np.array(self.endog)

            y_calibrated_ = inv_emb_nclass(self.y_calibrated,self.nclass-1,0.5)
            
            msa = acb.metrics_binary_classifier(y_endog_, y_calibrated_, self.misclassif)
            
        elif self.misclassif == False and self.nclass==2:

            y_ = np.array(self.y)

            y_estimated_ =  inv_emb_nclass(self.y_estimated,self.nclass-1,0.5)

            msa = acb.metrics_binary_classifier(y_, y_estimated_, self.misclassif,average='macro')

        elif self.misclassif ==True and self.nclass>2:

            y_endog_ = np.array(self.endog_)

            y_calibrated =np.argmax(self.y_calibrated,axis=1)
            
            msa = ac.metrics_multi_classifier(y_endog_, y_calibrated_, self.misclassif,None)
            
        elif self.misclassif == False and self.nclass>2:
         
            y_=np.array(self.y_)

            y_estimated_=np.argmax(self.y_estimated,axis=1)
            
            """
            with open('y','wb') as f:
                np.save(f,self.y)
                f.close()
            with open('yp','wb') as g:
                np.save(g,self.y_estimated_)
                g.close()
            """
            
            msa = ac.metrics_multi_classifier(y_, y_estimated_, self.misclassif,None)
           
        else:
            pass

        if self.misclassif == True:
            self.msb = msa
        else:
            self.msb = msa[:7]
        return self.msb

    def roc_curve(self):
        if self.nclass==2:
            fpr, tpr, _ = metrics.roc_curve(self.y, self.y_estimated)
        else:
            fpr, tpr = self.roc_curve_multiple()
        # Area Under Curve (AUC) of Receiver operating characteristic (ROC)
        # However AUC=1 indicating a perfect separation of the two classes
        auc = metrics.roc_auc_score(self.y, self.y_estimated)
        self.aucr = auc

        self.fpr = fpr
        self.tpr = tpr
        return self.fpr, self.tpr, self.aucr

    def roc_curve_multiple(self):
        fpr={}
        tpr={}
        aucr ={}
        for class_id in range(self.nclass):
            
            fpr[class_id],tpr[class_id],_=metrics.roc_curve(self.y_emb.iloc[:,class_id],self.y_estimated.iloc[:,class_id])
            aucr[class_id] = metrics.roc_auc_score(self.y_emb.iloc[:,class_id],self.y_estimated.iloc[:,class_id])
            
        return fpr, tpr,aucr

    def classification_report(self):
        return classification_report(self.y,self.y_estimated))

    def summary_models(self):
        self.model_sim=Sklearn_simula(self.model).fit(self)
        return(
            self.model_sim.fit_transf(self), #exog_fit_transform
            self.model_sim.transform(self),  #exog_transf
            self.model_sim.decision_function(self), #self.decision_on_class
            self.model_sim.calibrate(self), #self.y_calibrated
            self.model_sim.predict(self), #self.y_estimated
            self.model_sim.predict_proba(self), #self.y_estimated_proba
            self.model_sim.score(self). #self.score
            self.model_sim.get_params(self), #self.getparameters
            self.model_sim.confusion_matrix(self), #self.msb
            self.model_sim.roc_curve(self), #fpr,tpr,aucr
            self.model_sim.classification_report(self), #print
            )
class Sklearn_linear_filter(Sklearn_simula):
    """Simulation with the original data ."""

    def __init__(self, names,model, model_names,exog, endog, x, y,**kwargs):
#family, method, par_reg, task, mis_endog

        self.names = names
        self.model_names=model_names
        self.exog = exog
        self.model = model
        self.name = name
        self.endog = endog
        self.x = x
        self.y = y

        # parameters
        keys=["par","ntarget","nclass","missclassif","par_reg","columns"]
         
        try:       
            self.par = kwargs["par"]
        except:
            pass
        try:
            self.ntarget=kwargs["ntarget"]
        except:
            pass
        try:
            self.nclass=kwargs["nclass"]
        except:
            pass
        try:
            self.missclassif=kwargs["missclassif"]
        except:
            pass
        try:
            self.par_reg=kwargs["par_reg"]
        except:
            pass
        try:
            self.columns = kwargs["columns"]
        except:
            pass
        try:
            self.task=kwargs["task"]
        except:
            pass
        
    def sklearn_linear_classification(self):

        """Linear Methods to Classification."""
           
        to_model_names = []
        to_exog_fit_transf=OrderedDict()
        to_exog_transform=OrderedDict()
        to_decision_on_class=OrderedDict()
        to_y_calibrated = OrderedDict()
        to_y_estimated = OrderedDict()
        to_y_estimated_proba= OrderedDict()
        to_score=[]
        to_params=OrderedDict()
        to_fpr = OrderedDict()
        to_tpr = OrderedDict()
        to_aucr= OrderedDict()
        to_fp_index = OrderedDict()
        to_fn_index = OrderedDict()
        to_classification_report= OrderedDict()
 
        for name,clf in zip(self.names,self.model_names):
            self.name=name
            self.model=clf
            (
             exog_fit_transform,
             exog_transform,
             decision_on_class,
             y_calibrated,
             y_estimated,
             y_estimated_proba,
             score,
             getparameters,
             sc,
             fpr,tpr,aucr,
             classification_report
                
            ) =Sklearn_simula.summary_models(sef) 

            print(input("PAUSE"))
            
            to_model_names.append(self.name)
            to_exog_fit_transform[self.name]=exog_fit_transform
            to_exog_transform[self.name]=exog_transform
            to_decision_on_class[self.name]=decision_on_class
            to_y_calibrated[str(self.name)] = y_calibrated
            to_y_estimated[str(self.name)] = y_estimated
            to_y_estimated_proba[self.name]= y_estimated_proba
            to_score.append(score)
            to_params[self.name]=getparameters

            
            if self.misclassif == True:
                    confusion_matrix.append(sc[:8])
                    to_fp_index[str(self.name)] = sc[10]
                    to_fn_index[str(self.name)] = sc[11]
        
            else:
                    confusion_matrix.append(sc)
                    if self.nclass==2: 
                        to_fpr[str(self.name)] = fpr
                        to_tpr[str(self.name)] = tpr
                        to_aucr[str(self.name)] = aucr
                    elif self.nclass>2:
                        for class_id in range(self.nclass):
                            to_fpr[(str(self.name),class_id)] = fpr[class_id]
                            to_tpr[(str(self.name),class_id)] = tpr[class_id]
                    else:
                        pass
                        
        if self.nclass==2:
            if self.misclassif == True:
                FP_Index = pd.DataFrame.from_dict(to_fp_index, orient="index")
                FN_Index = pd.DataFrame.from_dict(to_fn_index, orient="index")
        
                confusion_matrix = pd.DataFrame(
                    confusion_matrix,
                    index=to_model_names,  # acc,TP,TN,FP,FN,BIAS,pt1,pp1,TNTP,total,fp_index,fn_index
                    columns=[
                        "ACC",
                        "TP",
                        "TN",
                        "FP",
                        "FN",
                        "BIAS",
                        "P[X=1]",
                        "P[X*=1]",
                    ],
                )
            else:
                to_fpr["AUC=0.5"] = fpr
                to_tpr["AUC=0.5"] = fpr
                FPR = pd.DataFrame.from_dict(to_fpr, orient="index")
                TPR = pd.DataFrame.from_dict(to_tpr, orient="index")
                confusion_matrix = pd.DataFrame(
                    confusion_matrix,
                    index=to_model_names,
                    columns=["ACC", "TPR", "TNR", "PPV", "FPR", "FNR", "DOR"],
                )
        else:
            pass
        #to_params = (
        #    np.array(to_params).reshape(len(to_model_names), len(self.columns)).T
        #)
        #params_table = pd.DataFrame(
        #    to_params, index=self.columns, columns=to_model_names
        #)

        y_calibrated_table = pd.DataFrame.from_dict(to_y_calibrated)

        y_estimated_table = pd.DataFrame.from_dict(to_y_estimated)

        y_estimated_proba_table = pd.DataFrame.from_dict(to_y_estimated_proba)

        #to_z_score = (
        #    np.array(to_z_score).reshape(len(to_model_names), len(self.columns) - 1).T
        #)
        #z_score_table = pd.DataFrame(
        #    to_z_score, index=self.columns[1:], columns=to_model_names
        #)
        mis_endog_table=[]
 
        if self.misclassif == True:
                return (
                    to_model_names,
                    y_calibrated_table,
                    y_estimated_table,
                    y_estimated_proba_table,
                    score_table,
                    FP_Index,
                    FN_Index,
                    params_table,
                    confusion_matrix,
                    mis_endog_table 
                )
        else:
                return (
                    to_model_names,
                    y_calibrated_table,
                    y_estimated_table,
                    y_estimated_proba_table,
                    score_table,
                    FPR,
                    TPR,
                    params_table,
                    confusion_matrix,
                    mis_endog_table
                    
                )
        
