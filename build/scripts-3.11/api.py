#!python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:07:28 2023

@author: delta
"""

__all__ = [
    """
    "Best_features_filter",
    "Best_features_wrap",
    "BinaryModel",
    "BinaryResults",
    "BinaryResultsWrapper",
    "Collocation_points_simula",
    "Collocation_simula",
    "Correlation",
    "DATA",
    "Data_Analytics",
    "Data_Engineering",
    "Data_Generation",
    "Data_Output",
    "Data_Quality",
    "Data_Visualisation",
    "DiscreteModelShape",
    "DiscreteResults",
    "Draw_binary_classification_results",
    "Draw_numerical_results",
    "Drawing2d",
    "Features_selection",
    "GenLogit",
    "GenLogitResults",
    "GenLogit_output",
    "Init",
    "L1BinaryResults",
    "L1BinaryResultsWrapper",
    "Logit",
    "LogitResults",
    "MDS",
    "MetricBinaryClassifier",
    "Multivariate_pdf",
    "MyDB",
    "Nothing",
    "OrderedResults",
    "PCA",
    "Pipeline",
    "SVD",
    "Simleng",
    "Simleng_strategies",
    "Sklearn_linear_filter",
    "Sklearn_simula",
    "Statsmodels_linear_filter",
    "Statsmodels_simula",
    "Table_results",
    "nincrease",
    "pw",
"""
]

# __version__ = versioneer.get_versions()["Version"]

all_names_import = [
    """   
 from sklearn.feature_selection import VarianceThreshold as skVT
 from sklearn.mixture import GaussianMixture
 from statsmodels.distributions import genpoisson_p
 import sys
 from resources.db import MyDB
 import statsmodels.api as sm
 from sklearn.ensemble import IsolationForest as skIF
 from sklearn.ensemble import AdaBoostRegressor as skABR
 import statsmodels.regression.linear_model as lm
 from statsmodels.stats.outliers_influence import variance_inflation_factor
 import warnings
 from statsmodels import base
 from sklearn.model_selection import StratifiedKFold
 from sklearn import metrics
 from sklearn.decomposition import PCA as skPCA
 from sklearn.neighbors import BallTree,NearestCentroid,NearestNeighbors, \
 from scipy.stats import multivariate_normal
 from sklearn.gaussian_process.kernels import RationalQuadratic as skRQ,ExpSineSquared as skESS, DotProduct as skDP, PairwiseKernel as skPK
 from sklearn.metrics import balanced_accuracy_score as balanced_acc
 from sklearn.ensemble import RandomForestRegressor as skRFR
 import statsmodels.discrete.discrete_model as smd
 from data_manager.feature_eng import Correlation
 import re
 from resources.sets import data_list_unpack_dict
 from data_manager.feature_eng import Best_features_filter
 from sklearn.ensemble import GradientBoostingRegressor as skGBR
 from resources.distributions import lump_dataset,cdf_lump
 from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod, abstractproperty
 from collections import OrderedDict
 from IPython import get_ipython
 from sklearn.linear_model import enet_path,lars_path,lasso_path, orthogonal_mp,orthogonal_mp_gram,ridge_regression
 from sklearn import semi_supervised
 from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, ClassifierChain
 from sklearn import manifold
 from sklearn.gaussian_process.kernels import RBF as skRBF,Matern as skMatern
 from sklearn import datasets
 from statsmodels.datasets import utils as du
 from data_manager.feature_eng import PCA
 from data_manager.quality import Data_Visualisation, Data_Analytics, Data_Quality 
 from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier,OutputCodeClassifier
 from statsmodels.tools.decorators import cache_readonly
 import sklearn
 from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
 import seaborn as sns
 from resources.sets import array_sort_y_on_x, data_list_to_matrix
 from subprocess import call,run
 from sklearn.feature_selection import SelectKBest as skSB
 from data_abstract import DATA
 from scipy import stats
 from sklearn import linear_model as sklm
 from resources.io import find_full_path
 from sklearn.model_selection import learning_curve as sklearncur,validation_curve as skvalcur
 from scipy.linalg import svd as sksvd
 from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron
 from supervised.metrics_classifier_statsmodels import MetricBinaryClassifier
 from pandas import MultiIndex, get_dummies
 from sklearn.feature_selection import RFECV as skRFECV
 from statsmodels.tools.numdiff import approx_fprime_cs
 import time
 from sklearn.linear_model import SGDClassifier, SGDRegressor, SquaredLoss, TheilSenRegressor
 from statsmodels.compat.pandas import Appender
 from output.table import Table_results
 from output.graphics import Draw_numerical_results, Draw_binary_classification_results
 from data_manager.quality import Data_Visualisation 
 from sklearn.linear_model import ARDRegression,BayesianRidge, ElasticNet,ElasticNetCV
 from sklearn.svm import SVC, NuSVC, SVR, NuSVR, OneClassSVM, LinearSVC, \
 from sklearn.kernel_approximation import RBFSampler as skRBFSampler,SkewedChi2Sampler as skSChi2Sampler,AdditiveChi2Sampler as skAdChi2Sampler,Nystroem as skNystroem
 from sklearn.ensemble import GradientBoostingClassifier as skGBC
 from sklearn.linear_model import LinearRegression, Log, LogisticRegression, LogisticRegressionCV
 from sklearn.pipeline import make_pipeline
 from scipy.special import digamma, gammaln, loggamma, polygamma
 from sklearn.svm import SVR as skSVR
 import pandas as pd
 from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB 
 from matplotlib import colors as mcolors
 from sklearn.feature_selection import mutual_info_classif as skmiC
 from   data_manager.feature_eng import Best_features_wrap
 from sklearn.semi_supervised import LabelPropagation, LabelSpreading
 from sklearn.linear_model import Hinge,Huber, HuberRegressor, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
 from smt.surrogate_models import rbf
 from collections import OrderedDict,defaultdict
 from sklearn.preprocessing import Binarizer,\
 from sklearn.model_selection import GridSearchCV as skGridSCV
 from supervised.metrics_classifier import MetricBinaryClassifier
 from data_manager.feature_eng import Data_Engineering,Correlation,PCA,SVD,Best_features_filter, Best_features_wrap
 from init import Init
 from data_manager.generation import Data_Generation
 import matplotlib.pyplot as plt
 from sklearn.feature_selection import mutual_info_regression as skmiR
 from resources.output import table
 from resources.sets import reduce_value_list_double_to_single_key
 from statsmodels.base.l1_slsqp import fit_l1_slsqp
 import os
 from statsmodels.discrete.discrete_model import BinaryModel,BinaryResults,BinaryResultsWrapper
 from sklearn.pipeline import Pipeline
 from sklearn.linear_model import RANSACRegressor, Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV
 from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
 from sklearn.feature_selection import f_classif as skf_c
 from sklearn.feature_selection import f_regression as skf_r
 from sklearn.ensemble import AdaBoostClassifier as skABC
 from resources.scrapping import read_txt_to_dict
 from sklearn.feature_selection import RFE as skRFE
 import numpy as np
 from resources.sets import x_add_list
 from supervised.simulation_statsmodels import Statsmodels_linear_filter
 import statsmodels.base.wrapper as wrap
 from sklearn.manifold import MDS as skMDS, smacof as sksmacof,Isomap as skIsomap,TSNE as skTSNE,spectral_embedding as skSpectEmbed
 from resources.io import find_full_path,file_reader
 from sklearn.ensemble import BaggingClassifier as skBagC
 from sklearn.ensemble import VotingClassifier as skVC
 from sklearn import svm
 from sklearn import metrics 
 from sklearn.gaussian_process import GaussianProcessClassifier as skGPC
 from resources.pandas import mapping_zero_one
 from mpl_toolkits.mplot3d import Axes3D
 from statsmodels.tools import data as data_tools, tools
 from sklearn.gaussian_process import GaussianProcessRegressor as skGPR
 from sklearn.kernel_ridge import KernelRidge as skKRidge
 from sklearn.model_selection import train_test_split
 from output.Draw_numerical_results import frame_from_dict
 from functools import wraps
 from biokit.viz import corrplot         
 from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as skQDA
 from statsmodels.multivariate.pca import PCA as smPCA
 from scipy import stats,linalg
 from sklearn.svm import SVC as skSVC
 from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
 from supervised.metrics_classifier_statsmodels import MetricBinaryClassifier as ac
 from sklearn.ensemble import RandomForestClassifier as skRFC
 from data_manager.quality import Data_Visualisation, Data_Analytics, Data_Quality
 from sklearn.ensemble import BaggingRegressor as skBagR
 from biokit.viz import corrplot
 from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,\
 from data_manager.feature_eng import Data_Engineering
 from sklearn import preprocessing
 from statsmodels.tools.sm_exceptions import (
 from resources.algebra import array_mapping_zero_one
 from sklearn.random_projection import BaseRandomProjection, GaussianRandomProjection, SparseRandomProjection
 import statsmodels.base.model as base
 from sklearn.isotonic import IsotonicRegression as skIR
 from sklearn.model_selection import cross_val_score
 from data_manager.quality import Data_Visualisation, Data_Analytics
 from scipy import special, stats
 from sklearn.neural_network import BernoulliRBM,MLPClassifier,MLPRegressor
 from sklearn.model_selection import KFold
 from data_manager.feature_eng import SVD
 from statsmodels.base.data import handle_data  # for mnlogit
 from resources.manipulation_data import data_add_constant,find_subsets_predictors
 from simula.strategies_features_selection import Features_selection
 from resources.manipulation_data import data_dummy_binary_classification
 from distutils.dir_util import copy_tree
 from tests.clsport import port
 from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
 from scipy.stats import nbinom
 from sklearn.feature_selection import SelectFromModel as skFM
 from resources.manipulation_data import data_add_constant
 from sklearn.linear_model import ModifiedHuber, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV
 from supervised import GenLogitclass
"""
]

__all__names__ = [
    """    
neighbors
statsmodels
naive_bayes
random_projection
init
base
output
data_abstract
compat
smt
linear_model
svm
multioutput
sklearn
pandas
algebra
simula
viz
manipulation_data
feature_eng
ensemble
distributions
resources
kernels
tests
sets
strategies_features_selection
mplot3d
l1_slsqp
Draw_numerical_results
clsport
linalg
decomposition
model_selection
datasets
semi_supervised
distutils
discrete_model
biokit
supervised
outliers_influence
matplotlib
multivariate
dir_util
mixture
pipeline
discrete
functools
abc
io
manifold
data
special
scipy
graphics
surrogate_models
neural_network
preprocessing
decorators
kernel_ridge
metrics_classifier
pca
kernel_approximation
db
isotonic
simulation_statsmodels
mpl_toolkits
stats
subprocess
data_manager
sm_exceptions
tree
metrics_classifier_statsmodels
generation
gaussian_process
table
quality
tools
feature_selection
numdiff
metrics
scrapping
IPython
collections
multiclass
discriminant_analysis
"""
]
import re
import sys

from simleng_ai import run_simleng_line


if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])
    sys.exit(run_simleng_line.main())
