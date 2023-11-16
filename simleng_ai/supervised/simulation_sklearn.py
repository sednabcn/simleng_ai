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


from sklearn.feature_selection import SelectKBest as skSB
from sklearn.feature_selection import f_classif as skf_c
from sklearn.feature_selection import f_regression as skf_r
from sklearn.feature_selection import VarianceThreshold as skVT
from sklearn.feature_selection import RFE as skRFE
from sklearn.feature_selection import RFECV as skRFECV
from sklearn.feature_selection import mutual_info_classif as skmiC
from sklearn.feature_selection import mutual_info_regression as skmiR
from sklearn.feature_selection import SelectFromModel as skFM

from sklearn.model_selection import (
    learning_curve as sklearncur,
    validation_curve as skvalcur,
)

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

colors = [ic for ic in mcolors.BASE_COLORS.values() if ic != (1, 1, 1)]


class nincrease:
    def fit_on_increasing_size(model):
        n_samples = 100
        n_features_ = np.arange(10, 800, 20)
        r2_train, r2_test, snr = [], [], []
        for n_feature in n_features_:
            # Sample the dataset (* 2 nb of samples)
            n_features_info = int(n_features / 10)
            np.random.seed(42)  # Make reproducible
            X = np.random.randn(n_samples * 2, n_features)
            beta = np.zeros(n_features)
            beta[:n_features_info] = 1
            Xbeta = np.dot(X, beta)
            eps = np.random.randn(n_samples * 2)
            y = Xbeta + eps
            # Split the dataset into train and test sample
            Xtrain, Xtest = X[:n_samples, :], X[n_samples:, :]
            ytrain, ytest = y[:n_samples], y[n_samples:]
            # fit/predict
            lr = model.fit(Xtrain, ytrain)
            y_pred_train = lr.predict(Xtrain)
            y_pred_test = lr.predict(Xtest)
            snr.append(Xbeta.std() / eps.std())
            r2_train.append(metrics.r2_score(ytrain, y_pred_train))
            r2_test.append(metrics.r2_score(ytest, y_pred_test))
            return n_features_, np.array(r2_train), np.array(r2_test), np.array(snr)

    def plot_r2_snr(n_features_, r2_train, r2_test, xvline, snr, ax):
        """
        Two scales plot. Left y-axis: train test r-squared. Right y-axis SNR.
        """
        ax.plot(n_features_, r2_train, label="Train r-squared", linewidth=2)
        ax.plot(n_features_, r2_test, label="Test r-squared", linewidth=2)
        ax.axvline(x=xvline, linewidth=2, color="k", ls="--")
        ax.axhline(y=0, linewidth=1, color="k", ls="--")
        ax.set_ylim(-0.2, 1.1)
        ax.set_xlabel("Number of input features")
        ax.set_ylabel("r-squared")
        ax.legend(loc="best")
        ax.set_title("Prediction perf.")
        ax_right = ax.twinx()
        ax_right.plot(n_features_, snr, "r-", label="SNR", linewidth=1)
        ax_right.set_ylabel("SNR", color="r")
        for tl in ax_right.get_yticklabels():
            tl.set_color("r")


class Pipeline:
    def pipeline():
        model = make_pipeline(preprocessing.StandardScaler(), sklm.LassoCV())
        # or
        # model = Pipeline([('standardscaler', preprocessing.StandardScaler()),
        #                  ('lassocv', sklm.LassoCV())])
        scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
        print("Test r2:%.2f" % scores.mean())

        print("== Linear regression: scaling is not required ==")
        model = sklm.LinearRegression()
        model.fit(X, y)
        print("Coefficients:", model.coef_, model.intercept_)
        print("Test R2:%.2f" % cross_val_score(estimator=model, X=X, y=y, cv=5).mean())
        print("== Lasso without scaling ==")
        model = sklm.LassoCV()
        model.fit(X, y)
        print("Coefficients:", model.coef_, model.intercept_)
        print("Test R2:%.2f" % cross_val_score(estimator=model, X=X, y=y, cv=5).mean())
        print("== Lasso with scaling ==")
        model = sklm.LassoCV()
        scaler = preprocessing.StandardScaler()
        Xc = scaler.fit(X).transform(X)
        model.fit(Xc, y)
        print("Coefficients:", model.coef_, model.intercept_)
        print("Test R2:%.2f" % cross_val_score(estimator=model, X=Xc, y=y, cv=5).mean())

    def pipeline_classification():
        # Datasets
        n_samples, n_features = 100, 100
        Cs = [20, 30, 50]
        X, y = datasets.make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=5, random_state=42
        )

        print("-- Remark: scaler is only done on outer loop --")
        print("-----------------------------------------------")
        lasso_cv = make_pipeline(
            ("standardscaler", preprocessing.StandardScaler()),
            ("lasso", sklm.LogisticRegressionCV(Cs=Cs, scoring=balanced_acc)),
        )
        # time
        scores = cross_val_score(estimator=lasso_cv, X=X, y=y, cv=5)
        print("Test bACC:%.2f" % scores.mean())
        print("=============================================")
        print("== Scaler + Elasticnet logistic regression ==")
        print("=============================================")
        print("----------------------------")
        print("-- Parallelize outer loop --")
        print("----------------------------")
        enet = Pipeline(
            [
                ("standardscaler", preprocessing.StandardScaler()),
                (
                    "enet",
                    sklm.SGDClassifier(
                        loss="log",
                        penalty="elasticnet",
                        alpha=0.0001,
                        l1_ratio=0.15,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        param_grid = {"enet__alpha": alphas, "enet__l1_ratio": l1_ratio}
        enet_cv = GridSearchCV(enet, cv=5, param_grid=param_grid, scoring=balanced_acc)
        # time
        scores = cross_val_score(
            estimator=enet_cv, X=X, y=y, cv=5, scoring=balanced_acc, n_jobs=-1
        )
        print("Test bACC:%.2f" % scores.mean())

    def pipeline_filters():
        np.random.seed(42)
        n_samples, n_features, n_features_info = 100, 100, 3
        X = np.random.randn(n_samples, n_features)

        beta = np.zeros(n_features)
        beta[:n_features_info] = 1
        Xbeta = np.dot(X, beta)
        eps = np.random.randn(n_samples)
        y = Xbeta + eps
        X[:, 0] *= 1e6  # inflate the first feature
        X[:, 1] += 1e6  # bias the second feature
        y = 100 * y + 1000  # bias and scale the output
        model = Pipeline(
            [
                ("anova", SelectKBest(f_regression, k=3)),
                ("sklm", sklm.LinearRegression()),
            ]
        )
        scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
        print("Anova filter + linear regression, test r2:%.2f" % scores.mean())
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            [
                ("standardscaler", preprocessing.StandardScaler()),
                ("lassocv", sklm.LassoCV()),
            ]
        )
        scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
        print("Standardize + Lasso, test r2:%.2f" % scores.mean())

    def pipeline_regression_cv():
        #!/usr/bin/env python

        # Regression pipelines with CV for parameters selection
        """
        Now we combine standardization of input features, feature selection and learner with
        hyper-parameter within a pipeline
        which is warped in a grid search procedure to select the best hyperparameters based
         on a (inner)CV. The overall is
        plugged in an outer CV.
        """
        # Datasets
        n_samples, n_features, noise_sd = 100, 100, 20
        X, y, coef = datasets.make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise_sd,
            n_informative=5,
            random_state=42,
            coef=True,
        )
        # Use this to tune the noise parameter such that snr < 5
        print("SNR:", np.std(np.dot(X, coef)) / noise_sd)
        print("=============================")
        print("== Basic linear regression ==")
        print("=============================")
        scores = cross_val_score(estimator=sklm.LinearRegression(), X=X, y=y, cv=5)
        print("Test r2:%.2f" % scores.mean())
        print("==============================================")
        print("== Scaler + anova filter + ridge regression ==")
        print("==============================================")

        t1 = time.time()
        anova_ridge = Pipeline(
            [
                ("standardscaler", preprocessing.StandardScaler()),
                ("selectkbest", SelectKBest(f_regression)),
                ("ridge", sklm.Ridge()),
            ]
        )
        param_grid = {
            "selectkbest__k": np.arange(10, 110, 10),
            "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        }
        t2 = time.time()
        print("CPU: %s" % str(t2 - t1))
        # Expect execution in ipython, for python remove the %time
        print("----------------------------")
        print("-- Parallelize inner loop --")
        print("----------------------------")

        t1 = time.time()
        anova_ridge_cv = GridSearchCV(
            anova_ridge, cv=5, param_grid=param_grid, n_jobs=-1
        )
        scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5)
        t2 = time.time()
        print("CPU: %s" % str(t2 - t1))

        print("Test r2:%.2f" % scores.mean())
        print("----------------------------")
        print("-- Parallelize outer loop --")
        print("----------------------------")

        t1 = time.time()
        anova_ridge_cv = GridSearchCV(anova_ridge, cv=5, param_grid=param_grid)
        scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5, n_jobs=-1)
        t2 = time.time()
        print("CPU: %s" % str(t2 - t1))

        print("Test r2:%.2f" % scores.mean())
        print("=====================================")
        print("== Scaler + Elastic-net regression ==")
        print("=====================================")
        alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        l1_ratio = [0.1, 0.5, 0.9]
        print("----------------------------")
        print("-- Parallelize outer loop --")
        print("----------------------------")

        t1 = time.time()
        enet = Pipeline(
            [
                ("standardscaler", preprocessing.StandardScaler()),
                ("enet", sklm.ElasticNet(max_iter=10000)),
            ]
        )
        param_grid = {"enet__alpha": alphas, "enet__l1_ratio": l1_ratio}
        enet_cv = GridSearchCV(enet, cv=5, param_grid=param_grid)
        scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5, n_jobs=-1)
        t2 = time.time()
        print("CPU: %s" % str(t2 - t1))

        print("Test r2:%.2f" % scores.mean())
        print("-----------------------------------------------")
        print("-- Parallelize outer loop + built-in CV--")
        print("-- Remark: scaler is only done on outer loop --")
        print("-----------------------------------------------")
        t1 = time.time()
        enet_cv = Pipeline(
            [
                ("standardscaler", preprocessing.StandardScaler()),
                (
                    "enet",
                    sklm.ElasticNetCV(max_iter=10000, l1_ratio=l1_ratio, alphas=alphas),
                ),
            ]
        )
        scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5)
        t2 = time.time()
        print("CPU: %s" % str(t2 - t1))

        print("Test r2:%.2f" % scores.mean())

####RETHINK_TO_MUCH######
class Sklearn_simula:
    """Simulation with statsmodels."""

    """
    x--X_test
    y--Y_test
    family--pdf to use in the model
    """

    def __init__(self, name, exog, endog, x, y, family, method, par_reg, task):
        # to be overridden in subclasses
        self.name = None
        self.model_name = None
        self.model = None
        self.exog = None
        self.endog = None
        self.x = None
        self.y = None
        self.columns = None
        self.family = None
        self.method = None
        self.par_reg = []
        self.task = None
        self.linear = False
        self.misclassif = False
        self.y_calibrated_new = None

    def fit(self):
        # change the name of the models
        if self.model_name == "sklm.OLS":
            print(self.model_name)
            self.model = sklm.OLS(self.endog, self.exog).fit()
        elif self.model_name == "sklm.LogisticRegression":
            print(self.model_name)
            self.model = sklm.LogisticRegression(self.endog, self.exog).fit()
        else:
            print("Error in loop")
        return self.model

    def fit_regularized(self):
        if self.method == "l1" and self.model_name == "smd.LogisticRegression":
            print(
                self.model_name
                + " Regularized wwith alpha: %.2f L1wt: %.2f " % (self.par_reg)
            )

            self.model = sklm.LogisticRegression(self.endog, self.exog).fit_regularized(
                method="l1", alpha=self.par_reg[0]
            )

        if self.method == "elastic_net" and self.model_name == "sklm.OLS":
            print(
                self.model_name
                + " Regularized wwith alpha: %.2f L1wt: %.2f " % (self.par_reg)
            )
            self.model = sklm.OLS(self.endog, self.exog).fit_regularized(
                method="elastic_net", alpha=self.par_reg[0], L1wt=self.par_reg[1]
            )

        return self.model

    def summary_models(self):
        self.z_score = self.model.params[1:] / self.model.bse[1:]
        return (
            self.model_name,
            self.model.summary(),
            self.model.params.values,
            self.model.resid_pearson,
            self.model.fittedvalues,
            self.model.bse,
            self.z_score.values,
        )

    def summary_LS_models(self):
        from statsmodels.sandbox.regression.predstd import wls_prediction_std  # ?

        if self.model_name == "sklm.OLS":
            self.prstd, self.iv_l, self.iv_u = wls_prediction_std(self.model)
        if self.model_name == "sklm.LogisticRegression":
            self.model.ssr = self.model.pearson_chi2
        return self.model.ssr, self.prstd, self.iv_l, self.iv_u

    def calibrate(self):
        self.y_calibrated = self.model.predict(self.exog)
        return self.y_calibrated

    def predict(self):
        if self.model_name == "sklm.LogisticRegression":
            self.y_estimated = self.model.predict(self.x, linear=self.linear)
        else:
            self.y_estimated = self.model.predict(self.x)
        return self.y_estimated

    def confusion_matrix(self):
        if self.misclassif == True:
            self.y_estimated_ = list(
                map(lambda x: np.where(x < 0.5, 0, 1), self.y_calibrated)
            )
            self.y = np.array(self.endog)
        else:
            self.y_estimated_ =  list(
                map(lambda x: np.where(x < 0.5, 0, 1), self.y_estimated)
            )
            self.y = np.array(self.y)

        self.y_estimated_ = np.array(self.y_estimated_)

        msa = ac.metrics_binary(self.y, self.y_estimated_, self.misclassif)
        if self.misclassif == True:
            self.msb = msa
        else:
            self.msb = msa[:7]
        return self.msb

    def roc_curve(self):
        fpr, tpr, _ = metrics.roc_curve(self.y, self.y_estimated)
        self.fpr = fpr
        self.tpr = tpr
        # Area Under Curve (AUC) of Receiver operating characteristic (ROC)
        # However AUC=1 indicating a perfect separation of the two classes
        auc = metrics.roc_auc_score(self.y, self.y_estimated)
        self.auc = auc

        return self.fpr, self.tpr, self.auc


class Sklearn_linear_filter(Sklearn_simula):
    """Simulation with the original data ."""

    def __init__(
        self, names, exog, endog, x, y, family, method, par_reg, task, mis_endog
    ):
        self.names = names
        self.exog = exog
        self.model = None
        self.endog = endog
        self.x = x
        self.y = y
        self.family = family
        self.method = method
        self.par_reg = par_reg  # list with two parameters
        self.task = task
        self.mis_endog = mis_endog
        self.misclassif = False

        if len(self.mis_endog) > 0:
            self.misclassif = True

    def sklearn_linear_supervised(self):
        """Linear Methods to Supervised prediction."""

        to_model_names = []
        to_params = []
        to_residuals = OrderedDict()
        confusion_matrix = []

        to_fitted_values = OrderedDict()
        to_iv_l = OrderedDict()
        to_iv_u = OrderedDict()
        to_z_score = []
        to_y_calibrated = OrderedDict()
        to_y_estimated = OrderedDict()
        to_fpr = OrderedDict()
        to_tpr = OrderedDict()
        to_fp_index = OrderedDict()
        to_fn_index = OrderedDict()
        to_mis_endog = OrderedDict()
        # change the name of models

        names
        sklm_models = ["OLS", "LogisticRegression"]
        for name in self.names:
            self.name = name
            if self.name in sklm_models:
                self.model_name = "sklm." + str(self.name)
            elif self.name in sklm_models:
                self.model_name = "sm." + str(self.name)
                self.linear = False
            else:
                print("Error in model_name selection")

            if self.name == "OLS" or "LogisticRegression":
                self.method = "elastic_net"
            else:
                self.method = "l1"
            self.columns = self.exog.columns

            if len(self.par_reg) == 0:
                self.model = Sklearn_simula.fit(self)
            else:
                self.model = Sklearn_simula.fit_regularized(self)

            (
                model_name,
                summary,
                params,
                resid_pearson,
                fitted_values,
                bse,
                z_score,
            ) = Sklearn_simula.summary_models(self)

            if self.task == "LinearRegression":
                ssr, prstd, iv_l, iv_u = Sklearn_simula.summary_LS_models(self)
                to_iv_l[str(self.model_name)] = iv_l
                to_iv_u[str(self.model_name)] = iv_u

            to_model_names.append(model_name)
            y_calibrated = Sklearn_simula.calibrate(self)
            y_estimated = Sklearn_simula.predict(self)

            to_y_calibrated[str(self.model_name)] = y_calibrated
            to_y_estimated[str(self.model_name)] = y_estimated
            to_params.append(params)
            to_residuals[str(self.model_name)] = resid_pearson
            to_fitted_values[str(self.model_name)] = fitted_values

            to_z_score.append(z_score)

            if self.task == "BinaryClassification":
                sc = Sklearn_simula.confusion_matrix(self)

                if self.misclassif == True:
                    confusion_matrix.append(sc[:8])
                    to_fp_index[str(self.model_name)] = sc[10]
                    to_fn_index[str(self.model_name)] = sc[11]
                    to_mis_endog[
                        str(self.model_name)
                    ] = Sklearn_simula.simple_corrected_mis_classification(self)

                else:
                    confusion_matrix.append(sc)
                    fpr, tpr = Sklearn_simula.roc_curve(self)
                    to_fpr[str(self.model_name)] = fpr
                    to_tpr[str(self.model_name)] = tpr

        if self.task == "BinaryClassification":
            if self.misclassif == True:
                FP_Index = pd.DataFrame.from_dict(to_fp_index, orient="index")
                FN_Index = pd.DataFrame.from_dict(to_fn_index, orient="index")
                LEN = [len(names) for names in to_mis_endog.values()]

                mis_endog_table = pd.DataFrame.from_dict(to_mis_endog, orient="index")

                mis_endog_table = mis_endog_table.T
                newindex = range(1, LEN[0] + 1)
                mis_endog_table = pd.DataFrame(
                    np.array(mis_endog_table),
                    index=newindex,
                    columns=mis_endog_table.columns,
                )
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

        elif self.task == "LinearRegression":
            iv_l_table = pd.DataFrame.from_dict(to_iv_l)
            iv_u_table = pd.DataFrame.from_dict(to_iv_u)
        else:
            pass
        to_params = (
            np.array(to_params).reshape(len(to_model_names), len(self.columns)).T
        )
        params_table = pd.DataFrame(
            to_params, index=self.columns, columns=to_model_names
        )

        residuals_table = pd.DataFrame.from_dict(to_residuals)

        fitted_values_table = pd.DataFrame.from_dict(to_fitted_values)

        y_calibrated_table = pd.DataFrame.from_dict(to_y_calibrated)

        y_estimated_table = pd.DataFrame.from_dict(to_y_estimated)

        to_z_score = (
            np.array(to_z_score).reshape(len(to_model_names), len(self.columns) - 1).T
        )
        z_score_table = pd.DataFrame(
            to_z_score, index=self.columns[1:], columns=to_model_names
        )

        if self.task == "BinaryClassification":
            if self.misclassif == True:
                return (
                    mis_endog_table,
                    y_calibrated_table,
                    y_estimated_table,
                    FP_Index,
                    FN_Index,
                    confusion_matrix,
                    to_model_names,
                )
            else:
                return (
                    y_calibrated_table,
                    y_estimated_table,
                    params_table,
                    residuals_table,
                    fitted_values_table,
                    z_score_table,
                    FPR,
                    TPR,
                    confusion_matrix,
                    to_model_names,
                )
        else:
            return (
                y_calibrated_table,
                y_estimated_table,
                params_table,
                residuals_table,
                fitted_values_table,
                z_score_table,
                iv_l_table,
                iv_u_table,
                to_model_names,
            )
