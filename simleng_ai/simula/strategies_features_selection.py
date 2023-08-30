import numpy as np
import pandas as pd
import re
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
from ..supervised.metrics_classifier_statsmodels import MetricBinaryClassifier
from ..output.table import Table_results
from ..output.graphics import Draw_numerical_results, Draw_binary_classification_results
from scipy import stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as skQDA

from collections import OrderedDict, defaultdict


class Features_selection(Data_Generation):
    def __init__(self, *args):
        self.idoc = -1
        self.data_train = args[0]
        self.data_test = args[1]
        self.data_dummy_train = args[2]
        self.data_dummy_test = args[3]
        self.proc = args[4]
        self.params = defaultdict()
        self.params = args[5]

        self.idoc = args[6]

        self.GenLogit_shape = self.params["GenLogit_shape"]
        self.index_columns_base = self.params["columns_search"]
        self.min_shuffle_size = int(self.params["min_shuffle_size"])
        self.subset_search = self.params["subset_search"]
        self.shuffle_mode = self.params["shuffle_mode"]
        self.filter_cond = self.params["filter_cond"]
        self.K_fold = int(self.params["K_fold"])

    def strategies_features_selection_master(self):
        print(self.proc)
        # print(self.data_train.values())
        # print(self.data_dummy_train.values())

        method_name = "simulation_" + str(self.proc)
        process = getattr(self, method_name, "Invalid Method selected")
        return process()

    def simulation_full_features(self):
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

        Data_Visualisation(self.idoc).data_head_tail(df)

        Data_Visualisation(self.idoc).data_feature_show(df)

        Data_Visualisation(self.idoc).data_features_show(df)

        Data_Analytics().data_describe(df)

        Data_Visualisation(self.idoc).data_features_draw_hist(U_train, 10)

        Correlation(U_train, df.columns, 0.9, self.idoc).correlation_training()
        #if self.idoc >= 1:
        #    print("Fig:Correlation matrices")
        Correlation(U_train, U_train.columns, 0.9, self.idoc).correlation_level()

        Best_features_filter(U_train, U_train.columns, 10).variance_influence_factors()

        names = ["GLM", "Logit", "GenLogit", "Probit"]
        exog = U_train_exog
        endog = V_train
        x = U_test_exog
        y = V_test
        par = self.GenLogit_shape
        family = "Binomial"
        method = ""
        par_reg = []
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
            names, exog, endog, x, y, par, family, method, par_reg, task, mis_classif
        ).statsmodels_linear_supervised()

        Table_results(
            0, y_calibrated_table, ".3f", "fancy_grid", "Validation", 60
        ).print_table()

        Table_results(
            0, y_estimated_table, ".3f", "fancy_grid", "Estimation", 60
        ).print_table()

        Table_results(
            0, params_table, ".3f", "fancy_grid", "Models Parameters", 60
        ).print_table()

        if self.idoc >= 1:
            print("Fig:Residuals of solvers on Training Data")

        Draw_numerical_results.frame_from_dict(
            residuals_table.index,
            residuals_table,
            "samples",
            "Residue Pearson",
            "Residuals of solvers on Training Data",
            "Log",
            " ",
            " ",
            "square",
            self.idoc,
        )

        Table_results(
            6, confusion_matrix, ".2f", "fancy_grid", "Confusion Matrix ", 60
        ).print_table()

        if self.idoc >= 1:
            print("Fig:Binary Classification using statsmodels")
        # FPR,TPR,model_names,params,x_test,y_test,y_predict,columns
        Title = "Binary Classification using Statsmodels"
        Draw_binary_classification_results(
            FPR,
            TPR,
            to_model_names,
            params_table,
            U_train_exog,
            V_train,
            U_test,
            V_test,
            y_calibrated_table,
            y_estimated_table,
            residuals_table,
            columns_train,
            Title,
            "",
            self.idoc,
        ).fpn_estimated()

        # Draw_binary_classification_results(FPR,TPR,to_model_names).\
        # fpn_estimated(V_test,y_estimated_table)
        if self.idoc >= 1:
            print("Fig:ROC_CURVE")
        to_model_names.append("AUC=0.5")
        Title = " ROC_CURVE "
        Draw_binary_classification_results(
            FPR,
            TPR,
            to_model_names,
            params_table,
            U_train_exog,
            V_train,
            U_test,
            V_test,
            y_calibrated_table,
            y_estimated_table,
            residuals_table,
            columns_train,
            Title,
            "",
            self.idoc,
        ).roc_curve()

    def simulation_features_selection_z_score(self):
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

        # Get the best two-predictors based on the p-values
        names = ["GLM", "Logit", "GenLogit", "Probit"]
        endog = V_train
        y = V_test
        par = self.GenLogit_shape
        family = "Binomial"
        method = ""
        par_reg = []
        task = "BinaryClassification"
        mis_classif = ""
        ACC_TWO = []
        NN = 0
        for name in columns_two_table:
            NN += 1
            columns_base = ["const"]
            columns_base.append(name[0])
            columns_base.append(name[1])
            exog = U_train_exog[columns_base]
            x = U_test_exog[columns_base]

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
                names,
                exog,
                endog,
                x,
                y,
                par,
                family,
                method,
                par_reg,
                task,
                mis_classif,
            ).statsmodels_linear_supervised()

            Table_results(
                5,
                confusion_matrix,
                ".2f",
                "fancy_grid",
                "Confusion Matrix :" + columns_base[1] + "," + columns_base[2],
                40,
            ).print_table()

            ACC_TWO.append(
                [columns_base[1], columns_base[2], confusion_matrix["ACC"].mean()]
            )

        ACC_TWO = pd.DataFrame(
            ACC_TWO, index=range(NN), columns=["predictor_1", "predictor_2", "ACC MEAN"]
        )
        Table_results(
            12, ACC_TWO, ".2f", "fancy_grid", "ACC MEAN for two predictors ", 30
        ).print_table()

        MAX_ACC_TWO = max(ACC_TWO["ACC MEAN"])
        Index_ACC_TWO = np.where(ACC_TWO["ACC MEAN"] == MAX_ACC_TWO)

        columns_two = []
        for ii in Index_ACC_TWO[0]:
            name1 = ACC_TWO.loc[ii, ACC_TWO.columns[0]]
            name2 = ACC_TWO.loc[ii, ACC_TWO.columns[1]]
            columns_two.append(name1)
            columns_two.append(name2)

        columns_Index = ["const"]
        for name in columns_two:
            columns_Index.append(str(name))

        columns_Index_plus = pd.Index(columns_Index)

        names = ["GLM", "Logit", "GenLogit", "Probit"]
        endog = V_train
        y = V_test
        par = self.GenLogit_shape
        family = "Binomial"
        method = ""
        par_reg = []
        task = "BinaryClassification"
        exog = U_train_exog[columns_Index_plus]
        x = U_test_exog[columns_Index_plus]
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
            names, exog, endog, x, y, par, family, method, par_reg, task, mis_classif
        ).statsmodels_linear_supervised()

        Table_results(
            5,
            confusion_matrix,
            ".2f",
            "fancy_grid",
            "Confusion Matrix :" + columns_Index_plus[1] + "," + columns_Index_plus[2],
            40,
        ).print_table()
        if self.idoc >= 1:
            print("Fig:Draw binary_classification_results of Best Predictors")
        # Draw_binary_classification_results of Best Predictors
        # based on p_values
        x = U_test_exog[columns_two]

        columns = columns_two.copy()
        columns.append(df.columns[-1])
        columns_two_copy = columns_two.copy()

        Title = (
            "Binary Classification Regions (Yes/No) using statsmodels"
            "over glu-ped plane"
        )
        Draw_binary_classification_results(
            FPR,
            TPR,
            to_model_names,
            params_table,
            U_train_exog,
            V_train,
            x,
            y,
            y_calibrated_table,
            y_estimated_table,
            residuals_table,
            columns,
            Title,
            "",
            self.idoc,
        ).draw_regions_binary_classification()

        print("INCREASING ADDING FEATURES AFTER Z-SCORE ANALYSIS")

        # Increase Features Method by add-features after Z-score selection
        names = ["Logit", "GenLogit"]
        method = ""
        par_reg = []
        cols_data = df.columns[:-1]
        mis_classif = ""

        (
            Iperfor,
            model_name_selected,
            columns_base,
            y_calibrated_table,
            y_estimated_table,
            pararesims_table,
            duals_table,
        ) = Best_features_wrap(self.idoc).add_feature_selection(
            names,
            cols_data,
            columns_two_copy,
            columns_Index_plus,
            U_train_exog,
            U_test_exog,
            V_train,
            V_test,
            "Binomial",
            method,
            par_reg,
            "BinaryClassification",
            90,
            1,
            mis_classif,
        )

        # Show mis-classification for the INCREASED BEST FEATURES SELECTION
        params = params_table.T
        kind = "Validation"
        Title = "Binary Classification using statsmodels" " (Validation Case)"
        if self.idoc >= 1:
            print("Fig:Draw binary Classification using Statsmodels(Validation Case)")
        Draw_binary_classification_results(
            FPR,
            TPR,
            names,
            params,
            U_train_exog,
            V_train,
            U_test_exog,
            V_test,
            y_calibrated_table,
            y_estimated_table,
            residuals_table,
            columns_base,
            Title,
            kind,
            self.idoc,
        ).draw_mis_classification()
        if self.idoc >= 1:
            print("Fig:Binary Classification(Prediction case)")
        kind = "Prediction"
        Title = "Binary Classification using statsmodels" " (Prediction Case)"
        Draw_binary_classification_results(
            FPR,
            TPR,
            names,
            params,
            U_train_exog,
            V_train,
            U_test_exog,
            V_test,
            y_calibrated_table,
            y_estimated_table,
            residuals_table,
            columns_base,
            Title,
            kind,
            self.idoc,
        ).draw_mis_classification()

    def simulation_K_fold_cross_validation(self):
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
        par = self.GenLogit_shape
        family = "Binomial"
        method = ""
        par_reg = []
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
            names, exog, endog, x, y, par, family, method, par_reg, task, mis_classif
        ).statsmodels_linear_supervised()

        Table_results(
            5, z_score_table, ".3g", "fancy_grid", "Z-score = params/bse", 30
        ).print_table()

        # Columns_two contain the best predictors
        columns_two, columns_Index = Best_features_wrap(self.idoc).z_score(
            z_score_table, 1.96
        )

        # Draw_binary_classification_results of Best Predictors
        # based on Z-score

        K_fold = self.K_fold
        names = ["Logit", "GenLogit"]
        method = ""
        par_reg = []
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
        ) = Best_features_wrap(self.idoc).cross_validation_binary_classification(
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

        Best_features_wrap(self.idoc).draw_K_fold_numerical_results(
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
        ) = Best_features_wrap(self.idoc).K_fold_numerical_results(
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
        ).draw_mis_classification()

    def simulation_pca(self):
        """Working to explore a spectral mehthod"""

        columns_pred = []
        columns_train, X_train, Y_train, df = data_list_unpack_dict(self.data_train)

        columns_test, X_test, Y_test, de = data_list_unpack_dict(self.data_test)

        U_train, V_train = data_list_unpack_dict(self.data_dummy_train)

        U_test, V_test = data_list_unpack_dict(self.data_dummy_test)

        n, m = U_train.shape

        PCA(U_train, n_components=m).pca()
        PCA(U_train, n_components=m).pca_draw_major_minor_factors(U_train)
        PCA(U_train, n_components=m).pca_show_table()
        PCA(U_train, n_components=m).pca_draw_by_components()

        # IT HAS BEEN DONE ONLY FOR ONE CASE:PIMAS--DELETE ONES
        # =========================================
        """
            list_index_col_subset=[int(num) for num in re.findall(r'\d+',self.index_columns_base)]

            subset_size_list=[int(num) for num in re.findall(r'\d+',self.subset_search)]

            min_subset_size=self.min_shuffle_size

            shuffle_mode=self.shuffle_mode

            filter_cond=self.filter_cond

            dict_params={}
            dict_params['list_index_col_subset']=list_index_col_subset
            dict_params['subset_size_list']=subset_size_list
            dict_params['min_subset_size']=min_subset_size
            
           
            """
        _, columns_ = self.find_subsets_predictors(U_train)

        # print(columns_)

        for ii in range(len(columns_)):
            columns_base = columns_[ii]

            print(columns_base)

            N_base = len(columns_base)
            U_train_reduced = U_train[columns_base]
            U_test_reduced = U_test[columns_base]

            U_train_pca = PCA(U_train_reduced, n_components=N_base).pca_transformation()
            U_test_pca = PCA(U_test_reduced, n_components=N_base).pca_transformation()
            U_train_pca_exog = data_add_constant(U_train_pca)
            U_test_pca_exog = data_add_constant(U_test_pca)

            # GET NEW PREDICTION WITH X-TRANSFORMED BY PCA
            names = ["Logit", "GenLogit"]
            endog = V_train.reset_index(drop=True)
            y = V_test
            par = self.GenLogit_shape
            family = "Binomial"
            method = ""
            par_reg = []
            task = "BinaryClassification"

            exog = U_train_pca_exog

            x = U_test_pca_exog

            mis_classif = ""
            # Observation with resampling based on perturbation
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
                names,
                exog,
                endog,
                x,
                y,
                par,
                family,
                method,
                par_reg,
                task,
                mis_classif,
            ).statsmodels_linear_supervised()

            Title = (
                "Confusion Matrix (PCA: var. in transformed space)"
                + " with "
                + str(N_base)
                + " predictors :"
                + columns_base[0]
            )
            for name in columns_base[1:]:
                Title += "," + name

            Table_results(
                6, confusion_matrix, ".2f", "fancy_grid", Title, 60
            ).print_table()

    def find_subsets_predictors(self, x):
        """Find subsets predictors in features selection methods"""
        """
                        Parameters
                        ----------
                        x: dataframe data
                        list_index_col_subset: list of indexes from the columns to search for a structured data
                        min_subset_size: minimum shuffle subset size to search
                        subset_size_list: list_of_subsets_to_aggregate
                        shuffle_mode:None,"combinations","permutations"
                        filter_cond: condition to filter subsets
     
                        Results
                        -------
                        subsets of indexes
                        
                        Notes
                        -----
                        
                        Examples
                        --------
    
                        """

        from itertools import combinations, permutations
        from ..resources.sets import order_mode

        df = x

        s_index = [int(num) for num in re.findall(r"\d+", self.index_columns_base)]

        k_search = [int(num) for num in re.findall(r"\d+", self.subset_search)]

        k_base = self.min_shuffle_size

        shuffle_mode = str(self.shuffle_mode)

        filter_cond = str(self.filter_cond)

        subsets_all = []

        if shuffle_mode == "combinations":
            subsets_all.extend(list(combinations(s_index, k_base)))

            if len(k_search) > 0:
                for k in k_search:
                    if k > k_base:
                        for item in list(combinations(s_index, k)):
                            subsets_all.append(item)

        elif shuffle_mode == "permutations":
            subsets_all.extend(list(permutations(s_index, k_base)))
            if len(k_search) > 0:
                for k in k_search:
                    if k > k_base:
                        for item in list(combinations(s_index, k)):
                            subsets_all.append(item)
        elif shuffle_mode == "None":
            subsets_all.extend(list(combinations(s_index, k_base)))
        else:
            pass
        # print(subsets_all)
        columns = [df.columns[list(item)].tolist() for item in subsets_all]
        # print(columns)
        # print(input("PAUSE"))
        if filter_cond == "same":
            columns = [item for item in columns if order_mode(df.columns, item)]
        else:
            pass
        # print(type(columns))
        # print(type(subsets_all))
        # print(columns)
        # print("=============================")
        # print(subsets_all)
        # print("=============================")

        return subsets_all, columns

    def simulation_additional_test(self):
        """
        TEST ADITIONAL OF features: npreg,glu,bmi,ped without PCA
        """
        columns_train, X_train, Y_train, df = data_list_unpack_dict(self.data_train)

        columns_test, X_test, Y_test, de = data_list_unpack_dict(self.data_test)

        U_train, V_train = data_list_unpack_dict(self.data_dummy_train)

        U_test, V_test = data_list_unpack_dict(self.data_dummy_test)

        cols_list = [int(num) for num in re.findall(r"\d+", self.index_columns_base)]
        columns = []
        for ii in cols_list:
            columns.append(df.columns[ii])

        columns_base = columns.copy()
        N_base = len(columns_base)
        U_train_test_exog = data_add_constant(U_train[columns_base])
        U_test_test_exog = data_add_constant(U_test[columns_base])

        names = ["Logit", "GenLogit"]
        endog = V_train.reset_index(drop=True)
        y = V_test
        par = self.GenLogit_shape
        family = "Binomial"
        method = ""
        par_reg = []
        task = "BinaryClassification"

        exog = U_train_test_exog.reset_index(drop=True)

        x = U_test_test_exog

        mis_classif = ""
        # Observation with resampling based on perturbation
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
            names, exog, endog, x, y, par, family, method, par_reg, task, mis_classif
        ).statsmodels_linear_supervised()

        Title = (
            "Confusion Matrix (original var.)"
            + " with "
            + str(N_base)
            + " predictors :"
            + columns_base[0]
        )
        for name in columns_base[1:]:
            Title += "," + name

        Table_results(6, confusion_matrix, ".2f", "fancy_grid", Title, 60).print_table()

    def simulation_discriminant_methods(self):
        columns_train, X_train, Y_train, df = data_list_unpack_dict(self.data_train)

        columns_test, X_test, Y_test, de = data_list_unpack_dict(self.data_test)

        U_train, V_train = data_list_unpack_dict(self.data_dummy_train)

        U_test, V_test = data_list_unpack_dict(self.data_dummy_test)

        # Manual selection of the best results in previous test

        cols_list = [[int(num) for num in re.findall(r"\d+", self.index_columns_base)]]

        columns_base = []
        for cols in cols_list:
            confusion_matrix = []
            columns_base = df.columns[cols]
            print(columns_base)
            N_base = len(columns_base)
            U_train_reduced = U_train[columns_base]
            U_test_reduced = U_test[columns_base]
            lda = skLDA(solver="svd", store_covariance=True)
            qda = skQDA(store_covariance=True)
            to_model_names = ["LDA", "QDA"]
            # ORIGINAL SPACE
            print("ORIGINAL VARIABLES")

            y_calibrated = lda.fit(U_train_reduced, V_train).predict(U_train_reduced)
            y_estimated_ = list(map(lambda x: np.where(x < 0.5, 0, 1), y_calibrated))
            y_t = np.array(V_train)
            y_est = np.array(V_test)
            (
                acc,
                tpr,
                tnr,
                ppv,
                fpr,
                fnr,
                dor,
                BIAS,
                pt1,
                pp1,
                _,
                _,
            ) = MetricBinaryClassifier.metrics_binary(y_t, y_estimated_, False)
            print("LDA_CALIBRATED")
            # print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
            y_estimated = lda.fit(U_test_reduced, V_test).predict(U_test_reduced)
            y_estimated_ = list(map(lambda x: np.where(x < 0.5, 0, 1), y_estimated))
            (
                acc,
                tpr,
                tnr,
                ppv,
                fpr,
                fnr,
                dor,
                BIAS,
                pt1,
                pp1,
                _,
                _,
            ) = MetricBinaryClassifier.metrics_binary(y_est, y_estimated_, False)
            print("LDA_ESTIMATED")
            # print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
            confusion_matrix.append([acc, tpr, tnr, ppv, fpr, fnr, dor, BIAS, pt1, pp1])
            # Quadratic Discriminant Analysis

            y_calibrated = qda.fit(U_train_reduced, V_train).predict(U_train_reduced)
            y_estimated_ = list(map(lambda x: np.where(x < 0.5, 0, 1), y_calibrated))
            (
                acc,
                tpr,
                tnr,
                ppv,
                fpr,
                fnr,
                dor,
                BIAS,
                pt1,
                pp1,
                _,
                _,
            ) = MetricBinaryClassifier.metrics_binary(y_t, y_estimated_, False)
            print("QDA_CALIBRATED")
            # print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
            y_estimated = qda.fit(U_test_reduced, V_test).predict(U_test_reduced)
            y_estimated_ = list(map(lambda x: np.where(x < 0.5, 0, 1), y_estimated))

            (
                acc,
                tpr,
                tnr,
                ppv,
                fpr,
                fnr,
                dor,
                BIAS,
                pt1,
                pp1,
                _,
                _,
            ) = MetricBinaryClassifier.metrics_binary(y_est, y_estimated_, False)
            print("QDA_ESTIMATED")
            # print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
            confusion_matrix.append([acc, tpr, tnr, ppv, fpr, fnr, dor, BIAS, pt1, pp1])
            # print("===========================================")
            confusion_matrix = pd.DataFrame(
                confusion_matrix,
                index=to_model_names,
                columns=[
                    "ACC",
                    "TPR",
                    "TNR",
                    "PPV",
                    "FPR",
                    "FNR",
                    "DOR",
                    "BIAS",
                    "P[X=1]",
                    "P[X*=1]",
                ],
            )
            Title = (
                "Confusion Matrix (original var.)"
                + " with "
                + str(N_base)
                + " predictors :"
                + columns_base[0]
            )
            for name in columns_base[1:]:
                Title += "," + name

            Table_results(
                6, confusion_matrix, ".2f", "fancy_grid", Title, 60
            ).print_table()
