#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:18:27 2018

@author: sedna
"""
from IPython import get_ipython

# get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.multivariate.pca import PCA as smPCA
from ..supervised.metrics_classifier import MetricBinaryClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

# from resources import Tools
from matplotlib import colors as mcolors

# colors for grahics with matplolib and plotly
from smt.surrogate_models import rbf
from ..resources.manipulation_data import data_dummy_binary_classification
from ..resources.sets import array_sort_y_on_x, data_list_to_matrix
from ..resources.pandas import mapping_zero_one
from ..resources.algebra import array_mapping_zero_one
from ..resources.distributions import lump_dataset, cdf_lump
from ..resources.output import table
from ..data_manager.feature_eng import Best_features_filter
from ..output.Draw_numerical_results import frame_from_dict

testsize = 0.4
# treshold=0.95
NDataSets = 3

features_treshold = [95.0]
# features_treshold=[90.0]
samples_transform = ["sum", "average", "optim"]
# samples_transform=['sum']

Datasets = ["Pima", "Sonar", "Ionosphere"]
Datasets = ["Ionosphere"]

ac = MetricBinaryClassifier()


class Collocation_simula:
    """Simulation_Collocation"""

    """
    |x--X_test
    |y--Y_test
    |family--pdf to use in the model
    |nbasis--Basis number
    |Based on https://smt.readthedocs.io/en/latest/_src_docs
    """

    def __init__(
        self, name, exog, endog, x, y, y_transf, family, method, par_reg, nbasis, task
    ):
        # to be overridden in subclasses
        self.name = None
        self.model_name = None
        self.model = None
        self.exog = None
        self.endog = None
        self.x = None
        self.y = None
        self.y_transf = None
        self.columns = None
        self.family = None
        self.method = None
        self.par_reg = []
        self.task = None
        self.linear = False
        self.misclassif = False
        self.y_calibrated_new = None
        self.n_basis = None
        self.z_train_pdf = None
        self.z_test_pdf = None

    def fit(self):
        if self.model_name == "rbf":
            self.model = rbf.RBF(d0=self.n_basis, poly_degree=0, print_global=False)
            self.model.set_training_values(self.exog, self.endog)
            self.model.train()
        return self.model

    def summary_models(self):
        return self.model_name, self.model.sol

    def calibrate(self):
        return self.model.predict_values(self.exog)

    def predict(self):
        return self.model.predict_values(self.x)

    def residuals(self):
        return np.abs(self.predict() - np.array(self.y).reshape(len(self.y), 1))

    def residuals_coll(self):
        return np.abs(
            self.predict() - np.array(self.y_transf).reshape(len(self.y_transf), 1)
        )

    def draw_prediction(self):
        import matplotlib.pyplot as plt

        xx, yy = array_sort_y_on_x(self.exog, self.endog, 0)
        xx4, yy4 = array_sort_y_on_x(self.x, self.predict(), 0)

        xx5, yy5 = array_sort_y_on_x(self.exog, self.calibrate(), 0)

        xx1, yy1 = array_sort_y_on_x(self.exog, self.z_train_pdf, 0)

        xx2, yy2 = array_sort_y_on_x(self.x, self.z_test_pdf, 0)

        xx3, yy3 = array_sort_y_on_x(self.x, self.residuals(), 0)

        xx6, yy6 = array_sort_y_on_x(self.x, self.residuals_coll(), 0)

        # the main axes is the subplot (1,1,1) by default
        plt.figure(figsize=(10, 8))
        plt.plot(xx, yy, "o")
        plt.plot(xx4, yy4, "r")
        plt.axis([0.0, 1.0, 0.0, 1.0])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Training data", "Prediction"], loc="center")
        plt.title(
            "LSC Method [Gauss. Var.:"
            + str(self.n_basis)
            + "]"
            + " [Transf :"
            + str(self.isamples)
            + "]"
            + " [PCA Best Features in "
            + str(self.data)
            + ":"
            + str(self.ifeatures)
            + "%]"
        )
        # this is an inset axes over the main axes
        plt.axes([0.2, 0.63, 0.2, 0.2], facecolor="y")
        plt.plot(xx, yy, "o")
        plt.plot(xx5, yy5, "g")
        plt.title("Validation")
        plt.legend(["Training data", "Validation data"])
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        # plt.xticks([])
        # plt.yticks([])
        # this is an other axes over the main axes
        plt.axes([0.5, 0.2, 0.35, 0.2], facecolor="#c7fdb5")

        m = 5
        scaley = 1
        for ii in range(0, len(xx1), m):
            xx = np.array([xx1[ii], xx1[ii]])
            yy = np.array([0.0, -scaley * yy1[ii]])
            plt.plot(
                xx1[ii],
                -scaley * yy1[ii],
                marker="o",
                markersize=6,
                linestyle="none",
                color="b",
                label="Training" if ii == 0 else "",
            )
            plt.plot(xx, yy, linewidth=1, color="b")

        for ii in range(0, len(xx2), m):
            xx = np.array([xx2[ii], xx2[ii]])
            yy = np.array([0.0, scaley * yy2[ii]])
            plt.plot(
                xx2[ii],
                scaley * yy2[ii],
                marker="o",
                markersize=6,
                linestyle="none",
                color="r",
                label="Testing" if ii == 0 else "",
            )
            plt.plot(xx, yy, linewidth=1, color="r")
        plt.plot(xx2, 0 * yy2, "k--")
        plt.title("Estimated Probability Density Functions ")
        plt.legend()
        plt.legend(loc="best")
        plt.xlim(0.0, 1.0)
        # plt.ylim(-2.0,2.0)
        plt.yticks([0.0])
        plt.ylabel("pdf")
        plt.xlabel("x")

        # this is an other axes over the main axes
        plt.axes([0.2, 0.35, 0.2, 0.2], facecolor="y")
        plt.plot(xx3, np.log(yy3), "r", label="Testing")
        plt.plot(xx6, np.log(yy6), "b", label=self.method)
        plt.title("Log(|Prediction Residuals|) ")
        plt.xlim(0.0, 1.0)
        # plt.ylim(0.,max(np.log(yy3)))

        # plt.yticks([])
        return plt.show()


class Collocation_points_simula(Collocation_simula):
    def __init__(
        self, names, exog, endog, x, y, y_transf, family, method, par_reg, nbasis, task
    ):
        # to be overridden in subclasses
        self.names = names
        self.model = None
        self.exog = exog
        self.endog = endog
        # np.transpose(np.array(endog))
        self.x = x
        self.y = y
        self.y_transf = y_transf
        # np.transpose(np.array(y))
        self.columns = None
        self.family = family
        self.method = method
        self.par_reg = []
        self.task = task
        self.linear = False
        self.misclassif = False
        self.y_calibrated_new = None
        self.n_basis = nbasis
        self.z_train_pdf = None
        self.z_test_pdf = None

    def collocation_1D_supervised(self):
        """1D Collocation Methods to Supervised Prediction"""
        to_model_names = []
        to_params = []

        for name in self.names:
            self.model_name = name

            if len(self.par_reg) == 0:
                self.model = Collocation_simula.fit(self)

            model_name, params = Collocation_simula.summary_models(self)
            to_model_names.append(model_name)
            to_params.append(params)
            y_calibrated = Collocation_simula.calibrate(self)
            y_estimated = Collocation_simula.predict(self)
            residuals = Collocation_simula.residuals(self)
            residuals_coll = Collocation_simula.residuals_coll(self)
        return (
            to_model_names,
            to_params,
            y_calibrated,
            y_estimated,
            residuals,
            residuals_coll,
        )

    def draw_prediction_collocation_1D(
        self, z_train_pdf, z_test_pdf, isamples, ifeatures, data
    ):
        """Draw behaviour of prediction using 1D collocation method"""
        to_model_names = []

        self.z_train_pdf = z_train_pdf
        self.z_test_pdf = z_test_pdf
        self.isamples = isamples
        self.ifeatures = ifeatures
        self.data = data
        for name in self.names:
            self.model_name = name

        if len(self.par_reg) == 0:
            self.model = Collocation_simula.fit(self)
        to_model_names.append(model_name)
        return Collocation_simula.draw_prediction(self)


AUC_MEAN = []
ACC_MEAN = []
FPFN_MEAN = []
AUC_TOP = []
ACC_TOP = []
FPFN_TOP = []

to_x_calibrated = defaultdict(dict)
to_x_estimated = defaultdict(dict)
to_y_calibrated = defaultdict(dict)
to_y_estimated = defaultdict(dict)
to_residuals = defaultdict(dict)

for data in Datasets:
    if data == "Pima":
        """Indias Pima Dataset"""

        pima_tr = pd.read_csv("pima.tr.csv", index_col=0)
        pima_te = pd.read_csv("pima.te.csv", index_col=0)

        # Training aata
        df = pima_tr

        # df.dropna(how="all", inplace=True) # drops the empty line at file-end
        columns = df.columns[:7]

        X_train = df[columns]
        Y_train = df["type"]

        # Testing data
        de = pima_te
        X_test = de[columns]
        Y_test = de["type"]

        # Mapping 'type' (Yes/No) to (1/0)
        V_train = data_dummy_binary_classification(Y_train, "No", 0, 1)
        V_test = data_dummy_binary_classification(Y_test, "No", 0, 1)

    elif data == "Sonar":
        sonar_all_data = pd.read_csv(
            "sonar.all-data.csv", index_col=0, header=None, names=range(0, 60)
        )

        y = sonar_all_data.pop(59).to_frame()
        X = sonar_all_data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=testsize, random_state=42, stratify=y
        )

        df = X_train
        de = X_test

        # Mapping 'type' (Yes/No) to (1/0)

        V_train = data_dummy_binary_classification(Y_train, "R", 0, 1)
        V_test = data_dummy_binary_classification(Y_test, "R", 0, 1)

    elif data == "Ionosphere":
        ionosphere_all_data = pd.read_csv("ionosphere_data_kaggle.csv", index_col=0)
        y = ionosphere_all_data.pop("label").to_frame()
        X = ionosphere_all_data
        X_transf = X.loc[:, X.columns[1:]]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_transf, y, test_size=testsize, random_state=42, stratify=y
        )
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_train["feature2"] = 0
        X_test = pd.DataFrame(X_test, columns=X.columns)
        X_test["feature2"] = 0

        df = X_train
        de = X_test
        # Mapping 'type' (Yes/No) to (1/0)
        V_train = data_dummy_binary_classification(Y_train, "b", 0, 1)
        V_test = data_dummy_binary_classification(Y_test, "b", 0, 1)
    else:
        pass

    # Best_PRedictors
    best_predictors_subset = []
    for indicator in features_treshold:
        treshold = indicator / 100.0
        if data == "Ionosphere":
            best_predictors_list = Best_features_filter.show_best_pca_predictors(
                X_transf, X_transf.shape[1], treshold
            )
        else:
            best_predictors_list = Best_features_filter.show_best_pca_predictors(
                X_train, X_train.shape[1], treshold
            )

        best_predictors_subset.append(best_predictors_list)

    U_train = mapping_zero_one(X_train)
    U_test = mapping_zero_one(X_test)

    Nfeat = len(features_treshold)
    Ntransf = len(samples_transform)
    Ncols = Nfeat * Ntransf

    index_var = [0.2, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0]
    index_var = [0.25, 0.3, 0.4, 0.5, 0.6, 2.0, 5.0, 7.5, 9.5, 10.0]

    par = 1.0

    N_var = len(index_var)

    index_columns = []
    for ii in range(Ncols):
        ifeatures, isamples = divmod(ii, Ntransf)

        index_columns.append(
            str(features_treshold[ifeatures]) + samples_transform[isamples][0]
        )

    auc_matrix = np.empty((N_var, Ncols))
    acc_matrix = np.empty((N_var, Ncols))
    fpfn_matrix = np.empty((N_var, Ncols)).astype("int")

    AUC = pd.DataFrame(auc_matrix, index=index_var, columns=index_columns)
    ACC = pd.DataFrame(acc_matrix, index=index_var, columns=index_columns)
    FPFN = pd.DataFrame(fpfn_matrix, index=index_var, columns=index_columns)

    for ii in range(Nfeat):
        predictors = best_predictors_subset[ii]

        ifeatures = features_treshold[ii]

        for jj in range(Ntransf):
            transf = samples_transform[jj]

            index_cols = index_columns[jj + ii * Ntransf]

            Confusion_Matrix = []

            # -Training Datasets
            U_train_exog = U_train[predictors]
            endog = V_train

            n, p = U_train.shape

            if transf == "optim":
                file = open("c.txt", "w")
                file.write(str(par))
                file.close()
                U_train_exog_t = sm.add_constant(U_train_exog, prepend=True)
                exog = U_train_exog_t
                model = smd.Logit(endog, exog).fit(disp=None, maxiter=100)
                par = model.params.values[1:]
            # --Transformations of training datasets

            U_train_transf = lump_dataset(U_train_exog, n, p, transf, par)

            U_train_transf_0_1 = array_mapping_zero_one(U_train_transf)

            (
                U_train_transf_0_1,
                endog,
                Likehood_train_pdf,
                Likehood_train_cdf,
            ) = cdf_lump(U_train_transf_0_1, n, endog, 0, 1)

            # -Testing datatsets
            m, p = U_test.shape
            U_test_exog = U_test[predictors]
            y_test = V_test

            # -Transformation of testing datasets
            U_test_transf = lump_dataset(U_test_exog, m, p, transf, par)

            U_test_transf_0_1 = array_mapping_zero_one(U_test_transf)

            U_test_transf_0_1, V_test, Likehood_test_pdf, Likehood_test_cdf = cdf_lump(
                U_test_transf_0_1, n, V_test, 0, 1
            )

            for var_ in index_var:
                to_x_calibrated[var_][index_cols] = U_train_transf_0_1
                to_x_estimated[var_][index_cols] = U_test_transf_0_1

                method = "RBF"

                # simula with collocation
                print(data + " Dataset ")
                print(
                    "Parameters:",
                    "PCA:",
                    str(features_treshold[ii]) + "%",
                    "\n",
                    "DATASET TRANSFORMATION:",
                    transf,
                    "\n",
                    "COLOCATION METHOD:",
                    method,
                )

                (
                    model_name,
                    params,
                    y_val_cdf,
                    y_pred_cdf,
                    residuals,
                    residuals_cdf,
                ) = Collocation_points_simula(
                    ["rbf"],
                    U_train_transf_0_1,
                    Likehood_train_cdf,
                    U_test_transf_0_1,
                    V_test,
                    Likehood_test_cdf,
                    "Binomial",
                    None,
                    "",
                    var_,
                    method,
                ).collocation_1D_supervised()

                # Drawing training and testing curves

                Collocation_points_simula(
                    ["rbf"],
                    U_train_transf_0_1,
                    Likehood_train_cdf,
                    U_test_transf_0_1,
                    V_test,
                    Likehood_test_cdf,
                    "Binomial",
                    None,
                    "",
                    var_,
                    method,
                ).draw_prediction_collocation_1D(
                    Likehood_train_pdf, Likehood_test_pdf, transf, ifeatures, data
                )

                # Calibrated

                to_y_calibrated[var_][index_cols] = y_val_cdf

                y_val = list(map((lambda x: np.where(x < 0.5, 0, 1)), y_val_cdf))

                auc = metrics.roc_auc_score(endog, y_val)
                auc = float_with_format(auc, 2) * 100
                msa = ac.metrics(endog, y_val)
                msb = [auc]

                msb.extend(float_list_with_format(msa[:9], 2))

                Confusion_Matrix.append(msb)

                # Testing

                to_y_estimated[var_][index_cols] = y_pred_cdf

                y_pred = list(map((lambda x: np.where(x < 0.5, 0, 1)), y_pred_cdf))

                auc = metrics.roc_auc_score(V_test, y_pred)

                auc = float_with_format(auc, 2) * 100
                msa = ac.metrics(V_test, y_pred)
                msb = [auc]
                msb.extend(float_list_with_format(msa[:9], 2))

                to_residuals[var_][index_cols] = residuals_cdf

                AUC.loc[var_, index_cols] = auc
                ACC.loc[var_, index_cols] = float_with_format(msa[0], 2)
                fpfn = (msa[6] + msa[8]) * 100.0 / len(V_test)
                FPFN.loc[var_, index_cols] = float_with_format(fpfn, 2)

                Confusion_Matrix.append(msb)

                Confusion_Matrix = pd.DataFrame(
                    Confusion_Matrix,
                    index=["Val", "Test" + str(index_cols[ii])],
                    columns=[
                        "AUC",
                        "ACC",
                        "TNTP",
                        "BIAS",
                        "PT=1",
                        "P=1",
                        "TP",
                        "FP",
                        "P=0",
                        "FN",
                    ],
                )

    print(Confusion_Matrix, "\n", "\n")

    # Table of Results
    residuals_table = pd.DataFrame.from_dict(to_residuals)
    x_calibrated_table = pd.DataFrame.from_dict(to_x_calibrated)
    y_calibrated_table = pd.DataFrame.from_dict(to_y_calibrated)
    x_estimated_table = pd.DataFrame.from_dict(to_x_estimated)
    y_estimated_table = pd.DataFrame.from_dict(to_y_estimated)

    table(AUC, ".2f", "fancy_grid", "AUC in " + data + " Dataset", 40)
    table(ACC, ".2f", "fancy_grid", "ACC in " + data + " Dataset", 40)
    table(FPFN, ".2f", "fancy_grid", "FPFN in " + data + " Dataset", 40)

    # Graphs of Results

    if Ncols < 8:
        frame_from_dict(
            AUC.index.values,
            AUC,
            "Gaussian variance values",
            "AUC",
            "AUC ROC CURVE " + data + " Dataset",
            "equal",
            False,
            "",
            "square",
            0,
        )
        frame_from_dict(
            ACC.index.values,
            ACC,
            "Gaussian variance values",
            "ACC(%)",
            "ACCURACY BINARY CLASSIFICATION " + data + " Dataset",
            "equal",
            False,
            "",
            "square",
            0,
        )
        frame_from_dict(
            FPFN.index.values,
            FPFN,
            "Gausssian variance values",
            "FP+FN(%)",
            "FP+FN BINARY CLASSIFICATION " + data + " Dataset",
            "equal",
            False,
            "",
            "square",
            0,
        )

    else:
        q, r = divmod(Ncols, 7)
        for ii in range(q + 1):
            if ii < q:
                AUC_split = AUC[AUC.columns[ii * 7 : (ii + 1) * 7]]

                ACC_split = ACC[ACC.columns[ii * 7 : (ii + 1) * 7]]

                FPFN_split = FPFN[FPFN.columns[ii * 7 : (ii + 1) * 7]]
            else:
                AUC_split = AUC[AUC.columns[q * 7 : Ncols + 1]]

                ACC_split = ACC[ACC.columns[q * 7 : Ncols + 1]]

                FPFN_split = FPFN[FPFN.columns[q * 7 : Ncols + 1]]

            frame_from_dict(
                AUC.index.values,
                AUC_split,
                "Gaussian variance values",
                "AUC",
                "AUC ROC CURVE " + data + " Dataset",
                "equal",
                False,
                "",
                "square",
                1,
            )
            frame_from_dict(
                ACC.index.values,
                ACC_split,
                "Gaussian variance values",
                "ACC(%)",
                "ACCURACY BINARY CLASSIFICATION " + data + " Dataset",
                "equal",
                False,
                "",
                "square",
                1,
            )
            frame_from_dict(
                FPFN.index.values,
                FPFN_split,
                "Gaussian variance values",
                "FP+FN(%)",
                "FP+FN BINARY CLASSIFICATION " + data + " Dataset",
                "equal",
                False,
                "",
                "square",
                1,
            )

    # compute mean values
    AUC_m = np.transpose(AUC).mean()
    AUC_m = np.array(AUC_m).reshape((1, len(index_var)))
    AUC_MEAN.append(AUC_m)
    AUC_m = pd.DataFrame(AUC_m, index=["AUC MEAN"], columns=index_var)
    table(AUC_m, ".2f", "fancy_grid", "AUC-MEAN ROC CURVE in " + data + " Dataset", 40)

    ACC_m = np.transpose(ACC).mean()
    ACC_m = np.array(ACC_m).reshape((1, len(index_var)))
    ACC_MEAN.append(ACC_m)
    ACC_m = pd.DataFrame(ACC_m, index=["ACC MEAN"], columns=index_var)
    table(ACC_m, ".2f", "fancy_grid", "ACC-MEAN in " + data + " Dataset", 40)

    FPFN_m = np.transpose(FPFN).mean()
    FPFN_m = np.array(FPFN_m).reshape((1, len(index_var)))
    FPFN_MEAN.append(FPFN_m)
    FPFN_m = pd.DataFrame(FPFN_m, index=["FPFN MEAN"], columns=index_var)
    table(FPFN_m, ".2f", "fancy_grid", "FPFN-MEAN in " + data + " Dataset", 40)

    # Compute top values

    AUC_t = np.transpose(AUC).max()
    AUC_t = np.array(AUC_t).reshape((1, len(index_var)))
    AUC_TOP.append(AUC_t)
    AUC_t = pd.DataFrame(AUC_t, index=["AUC TOP"], columns=index_var)
    table(
        AUC_t,
        ".2f",
        "fancy_grid",
        "AUC-TOP ROC CURVE [TOP] in " + data + " Dataset",
        40,
    )

    ACC_t = np.transpose(ACC).max()
    ACC_t = np.array(ACC_t).reshape((1, len(index_var)))
    ACC_TOP.append(ACC_t)
    ACC_t = pd.DataFrame(ACC_t, index=["ACC TOP"], columns=index_var)
    table(ACC_t, ".2f", "fancy_grid", "ACC-TOP in " + data + " Dataset", 40)

    FPFN_t = np.transpose(FPFN).min()
    FPFN_t = np.array(FPFN_t).reshape((1, len(index_var)))
    FPFN_TOP.append(FPFN_t)
    FPFN_t = pd.DataFrame(FPFN_t, index=["FPFN MINIMUM"], columns=index_var)
    table(FPFN_t, ".2f", "fancy_grid", "FPFN-MINIMUM in " + data + " Dataset", 40)


# MEAN VALUES

AUC_MEAN = data_list_to_matrix(AUC_MEAN, [len(index_var), len(Datasets)])

AUC_MEAN = pd.DataFrame(AUC_MEAN, index=index_var, columns=Datasets)

ACC_MEAN = data_list_to_matrix(ACC_MEAN, [len(index_var), len(Datasets)])

ACC_MEAN = pd.DataFrame(ACC_MEAN, index=index_var, columns=Datasets)

FPFN_MEAN = data_list_to_matrix(FPFN_MEAN, [len(index_var), len(Datasets)])


FPFN_MEAN = pd.DataFrame(FPFN_MEAN, index=index_var, columns=Datasets)


frame_from_dict(
    index_var,
    AUC_MEAN,
    "Gaussian variance values",
    "AUC-MEAN",
    "AUC-MEAN ROC CURVE",
    "equal",
    False,
    "",
    "square",
    0,
)

frame_from_dict(
    index_var,
    ACC_MEAN,
    "Gaussian variance values",
    "ACC-MEAN(%)",
    "ACCURACY BINARY CLASSIFICATION (AVERAGE VALUES)",
    "equal",
    False,
    "",
    "square",
    0,
)
frame_from_dict(
    index_var,
    FPFN_MEAN,
    "Gaussian variance values",
    "FP+FN MEAN(%)",
    "FP+FN BINARY CLASSIFICATION (AVERAGE VALUES) ",
    "equal",
    False,
    "",
    "square",
    0,
)

# TOP VALUES
AUC_TOP = data_list_to_matrix(AUC_TOP, [len(index_var), len(Datasets)])

AUC_TOP = pd.DataFrame(AUC_TOP, index=index_var, columns=Datasets)

ACC_TOP = data_list_to_matrix(ACC_TOP, [len(index_var), len(Datasets)])

ACC_TOP = pd.DataFrame(ACC_TOP, index=index_var, columns=Datasets)

FPFN_TOP = data_list_to_matrix(FPFN_TOP, [len(index_var), len(Datasets)])


FPFN_TOP = pd.DataFrame(FPFN_TOP, index=index_var, columns=Datasets)


frame_from_dict(
    index_var,
    AUC_TOP,
    "Gaussian variance values",
    "AUC-MAX",
    "AUC ROC CURVE (TOP VALUES)",
    "equal",
    False,
    "",
    "square",
    0,
)

frame_from_dict(
    index_var,
    ACC_TOP,
    "Gaussian variance values",
    "ACC-MAX(%)",
    "ACCURACY BINARY CLASSIFICATION (TOP VALUES) ",
    "equal",
    False,
    "",
    "square",
    0,
)
frame_from_dict(
    index_var,
    FPFN_TOP,
    "Gaussian variance values",
    "FP+FN MIN(%)",
    "FP+FN BINARY CLASSIFICATION (MIN VALUES) ",
    "equal",
    False,
    "",
    "square",
    0,
)
