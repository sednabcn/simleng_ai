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

from ..simula.strategies_statsmodels import Features_selection_statsmodels

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
        self.lib= args[6]
        self.idoc = args[7]

        self.GenLogit_shape = self.params["GenLogit_shape"]
        self.index_columns_base = self.params["columns_search"]
        self.min_shuffle_size = int(self.params["min_shuffle_size"])
        self.subset_search = self.params["subset_search"]
        self.shuffle_mode = self.params["shuffle_mode"]
        self.filter_cond = self.params["filter_cond"]
        self.K_fold = int(self.params["K_fold"])

    def strategies_features_selection_master(self):
        print(self.proc)
              
        pepe = [
                self.data_train,
                self.data_test,
                self.data_dummy_train,
                self.data_dummy_test,
                self.proc,
                self.params,
                self.lib, # including library
                self.idoc,               
            ]

        
        if self.lib=="stats":
            return Features_selection_statsmodels(*pepe).features_selection_statsmodels_master()
        elif self.lib=="sklearn":
            pass
        else:
            pass
