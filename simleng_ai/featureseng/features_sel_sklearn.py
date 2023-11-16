import numpy as np
import pandas as pd
import re
from distutils.util import strtobool
from ..data_manager.generation import Data_Generation
from ..data_manager.feature_eng import (
    Data_Engineering,
    Correlation,
    PCA,
    SVD,
    Best_features_filter,
    Best_features_wrap,
)
from ..resources.sets import data_list_unpack_dict
from ..resources.manipulation_data import data_add_constant, find_subsets_predictors

from ..data_manager.quality import Data_Visualisation, Data_Analytics
from ..supervised.simulation_statsmodels import Statsmodels_linear_filter

from ..output.table import Table_results
from ..output.graphics import Draw_numerical_results, Draw_binary_classification_results
from scipy import stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as skQDA

from collections import OrderedDict, defaultdict


class Features_selection_sklearn(Data_Generation):
    def __init__(self, *args):
        self.idoc = -1
        self.dataset=args[0]
        self.data_train = args[1]
        self.data_test = args[2]
        self.data_dummy_train = args[3]
        self.data_dummy_test = args[4]
        self.target =args[5]
        self.params = defaultdict()
        self.params = args[6]
        self.action = args[7]

        if self.target["GOAL"]=="CLASSIFICATION":
                       self.proc=self.action["method"]
                       self.lib=self.action["library"]
                       self.idoc=self.action["idoc"]
                       self.make_task=strtobool(str(self.action["make_task"]))
                       self.nclass=int(self.target["NCLASS"])
                       self.regpar=self.target["REGPAR"]
                       self.missclass=self.target["MISSCLASS"]
                       self.GenLogit_shape = self.params["GenLogit_shape"]
                       self.index_columns_base = self.params["columns_search"]
                       self.min_shuffle_size = int(self.params["min_shuffle_size"])
                       self.subset_search = self.params["subset_search"]
                       self.shuffle_mode = self.params["shuffle_mode"]
                       self.filter_cond = self.params["filter_cond"]
                       self.K_fold = int(self.params["K_fold"])
                       self.dataset_name=self.dataset["DATASET"]
        else:
             pass
                
        
        self.par=[]
        # PARAMETERS TO INCLUDE IN SUPERVISED METHODS
        if  isinstance(self.GenLogit_shape,float) or isinstance(self.GenLogit_shape,int):    
            self.par.append(self.GenLogit_shape)
        elif isinstance(self.GenLogit_shape,str):
            self.par.append(float(self.GenLogit_shape))
        if isinstance(self.nclass,int):
            self.par.append(self.nclass)
        
    def features_selection_sklearn_master(self):
        print(self.proc)
        # print(self.data_train.values())
        # print(self.data_dummy_train.values())
        method_name = "selection_" + str(self.proc)
        print(method_name)
        process = getattr(self, method_name, "Invalid Method selected")
        return process()

    def selection_full_features(self):
        if self.idoc == 1:
            """To create header of Report..."""
            pass
        """Working will full data to proof."""

        columns_train, X_train, Y_train, df = data_list_unpack_dict(self.data_train)

        columns_test, X_test, Y_test, de = data_list_unpack_dict(self.data_test)

        U_train, V_train = data_list_unpack_dict(self.data_dummy_train)

        U_test, V_test = data_list_unpack_dict(self.data_dummy_test)

        # addition a constant to compute intercept to apply \
        # statsmodels fitting

        U_train_exog = data_add_constant(U_train)
        U_test_exog = data_add_constant(U_test)

        Data_Visualisation(self.idoc,self.dataset_name).data_head_tail(df)

        Data_Visualisation(self.idoc,self.dataset_name).data_feature_show(df)

        Data_Visualisation(self.idoc,self.dataset_name).data_features_show(df)

        Data_Analytics(self.dataset_name).data_describe(df)

        Data_Visualisation(self.idoc,self.dataset_name).data_features_draw_hist(U_train, 10)

        Correlation(U_train, df.columns, 0.9, self.idoc).correlation_training()
        
        Correlation(U_train, U_train.columns, 0.9, self.idoc).correlation_level()

        Best_features_filter(U_train, U_train.columns, 10).variance_influence_factors()
    
    def selection_features_z_score(self):
        """Working with diferents criterie of features selection."""

        columns_train, X_train, Y_train, df = data_list_unpack_dict(self.data_train)

        columns_test, X_test, Y_test, de = data_list_unpack_dict(self.data_test)

        U_train, V_train = data_list_unpack_dict(self.data_dummy_train)

        U_test, V_test = data_list_unpack_dict(self.data_dummy_test)

        # Addition a constant to compute intercept to apply \
        #    regressoin models

        U_train_exog = data_add_constant(U_train)
        U_test_exog = data_add_constant(U_test)

        # Get two predictors based on p_values using scipy.ks_2samp
        z_score = {}
        z_score_table = {}
        columns = df.columns[:-1]

        for ii in range(len(columns)):
            for jj in range(len(columns)):
                if ii < jj:
                    columns_base = []
                    columns_base.append(columns[ii])
                    columns_base.append(columns[jj])
                    exog = U_train_exog[columns_base]
                    D_stats, p_value = stats.ks_2samp(
                        exog[columns_base[0]], exog[columns_base[1]]
                    )
                    z_score[p_value] = columns_base

        z_score_table_keys = sorted(z_score.keys(), reverse=False)
        columns_two_table = []
        for ii in z_score_table_keys:
            z_score_table[ii] = z_score[ii]
            columns_two_table.append(z_score[ii])
        z_score_table = pd.DataFrame.from_dict(z_score_table).T
        # Show the z_score of all predictors
        Table_results(
            5, z_score_table, ".3E", "fancy_grid", "p_values vs predictors", 15
        ).print_table()

        
    def selection_K_fold_cross_validation(self):
        print("K_fold CROSS-VALIDATION PROCESS WITH FITTING AND PREDICT")
        """
            | Best selection of features based on K_fold Cross-Validation
            | K_fold Cross-Validation + Add-features with Best_Features Selection
            | Fitting with K_Fold splitting
            """
        columns_train, X_train, Y_train, df = data_list_unpack_dict(self.data_train)

        columns_test, X_test, Y_test, de = data_list_unpack_dict(self.data_test)

        U_train, V_train = data_list_unpack_dict(self.data_dummy_train)

        U_test, V_test = data_list_unpack_dict(self.data_dummy_test)

        # Addition a constant to compute intercept to apply \
        # regressoin models

        U_train_exog = data_add_constant(U_train)
        U_test_exog = data_add_constant(U_test)

        # Fitting to wrap the best predictors
        names = ["GLM", "Logit", "GenLogit", "Probit"]
        exog = U_train_exog
        endog = V_train
        x = U_test_exog
        y = V_test
        par_ =self.par 
        family = "Binomial"
        method = ""
        par_reg =self.regpar
        task = "BinaryClassification"
        mis_classif = ""
        (
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
        ) = Statsmodels_linear_filter(
            names, exog, endog, x, y, par_, family, method, par_reg, task, mis_classif
        ).statsmodels_linear_supervised()

        Table_results(
            5, z_score_table, ".3g", "fancy_grid", "Z-score = params/bse", 30
        ).print_table()

        # Columns_two contain the best predictors
        columns_two, columns_Index = Best_features_wrap(self.idoc,self.GenLogit_shape).z_score(
            z_score_table, 1.96
        )

        # Draw_binary_classification_results of Best Predictors
        # based on Z-score

        K_fold = self.K_fold
        names = ["Logit", "GenLogit"]
        method = ""
        par_reg =self.regpar
        cols_data = df.columns[:-1]
        N_cols_base = len(columns_two)
        params = params_table.T
        columns_Index_plus = ["const"]
        for name in columns_Index:
            columns_Index_plus.append(name)
        params = params[columns_Index_plus]
        params = params.T
        x = U_test[columns_Index]
        y = V_test

        columns = columns_two.copy()
        columns.append(df.columns[-1])
        columns_two_two = columns_two.copy()
        mis_classif = ""

        (
            K_fold,
            N_features,
            data_fold,
            Iperformance,
        ) = Best_features_wrap(self.idoc,self.GenLogit_shape).cross_validation_binary_classification(
            names,
            cols_data,
            columns_two_two,
            columns_Index_plus,
            U_train_exog,
            U_test_exog,
            endog,
            y,
            family,
            method,
            par_reg,
            task,
            90,
            K_fold,
            mis_classif,
        )

        # Draw ACC and PPV from K_fold Cross-Validation fitting numerical results
        # print(K_fold,N_features)
        # print(Iperformance.keys())
        # for ii in range(K_fold):
        #        print(Iperformance[ii].keys(),Iperformance[ii].values())
        # print(input("STOP"))

        Best_features_wrap(self.idoc,self.GenLogit_shape).draw_K_fold_numerical_results(
            K_fold, N_cols_base, N_features, data_fold, Iperformance
        )

        # Generation of numerical results to draw mis-classification with K_fold splitting
        (
            Iperfor,
            model_name_selected,
            columns_base,
            x_train_fold_exog,
            y_train_fold,
            x_test_fold_exog,
            y_test_fold,
            y_calibrated_table,
            y_estimated_table,
            params_table,
            residuals_table,
        ) = Best_features_wrap(self.idoc,self.GenLogit_shape).K_fold_numerical_results(
            K_fold, N_cols_base, N_features, data_fold, Iperformance, 90
        )

        # Checking that the best reults are shown
        # Relation between Residues and misclassification in calibration stage
        params = params_table.T

        kind = "Validation"
        Title = (
            "K_fold Cross-Validation in Binary Classification using statsmodels"
            " (Validation Case)"
        )
        if self.idoc >= 1:
            print("Fig:Binary classification with K-fold validation")
        Draw_binary_classification_results(
            "",
            "",
            names,
            params,
            x_train_fold_exog,
            y_train_fold,
            x_test_fold_exog,
            y_test_fold,
            y_calibrated_table,
            y_estimated_table,
            residuals_table,
            columns_base,
            Title,
            kind,
            self.idoc,
            self.GenLogit_shape,
        ).draw_mis_classification()

        kind = "Prediction"
        Title = (
            "K_fold Cross-Validation in Binary Classification using statsmodels"
            " (Prediction Case)"
        )
        if self.idoc >= 1:
            print("Fig:Binary Missclassification case")
        Draw_binary_classification_results(
            "",
            "",
            names,
            params,
            x_train_fold_exog,
            y_train_fold,
            x_test_fold_exog,
            y_test_fold,
            y_calibrated_table,
            y_estimated_table,
            residuals_table,
            columns_base,
            Title,
            kind,
            self.idoc,
            self.GenLogit_shape,
        ).draw_mis_classification()

    def selection_with_pca(self):
        """Working to explore a spectral mehthod"""

        columns_pred = []
        columns_train, X_train, Y_train, df = data_list_unpack_dict(self.data_train)

        columns_test, X_test, Y_test, de = data_list_unpack_dict(self.data_test)

        U_train, V_train = data_list_unpack_dict(self.data_dummy_train)

        U_test, V_test = data_list_unpack_dict(self.data_dummy_test)

        n, m = U_train.shape

        PCA(U_train,m,self.idoc).pca()
        PCA(U_train,m,self.idoc).pca_draw_major_minor_factors(U_train)
        PCA(U_train,m,self.idoc).pca_show_table()
        PCA(U_train,m,self.idoc).pca_draw_by_components()

        
        parameters=[self.index_columns_base,self.subset_search,self.min_shuffle_size,\
                    self.shuffle_mode,self.filter_cond]
 
        
        _, columns_ = self.find_subsets_predictors(U_train,*parameters)

        # print(columns_)
 
