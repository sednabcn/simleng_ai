import numpy as np
import pandas as pd
import re
#===
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

#===
from scipy import stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as skQDA

from collections import OrderedDict, defaultdict

from ..simula.model_sel_sklearn import Model_selection_sklearn
 
# Make new one, PLEASE OCTOBER 14,2023
class Model_selection(Data_Generation):
    def __init__(self, *args):
        self.idoc = -1
        self.dataset =args[0]
        self.data_train = args[1]
        self.data_test = args[2]
        self.data_dummy_train = args[3]
        self.data_dummy_test = args[4]
        self.target =args[5]
        self.params = defaultdict()
        self.params = args[6]
        self.action = args[7]
                
    def strategies_model_selection_master(self):
         #print(self.action["method"])
          
         plist = [
                self.dataset,
                self.data_train,
                self.data_test,
                self.data_dummy_train,
                self.data_dummy_test,
                self.target,
                self.params,
                self.action,                                
            ]

         self.lib = self.action["library"]
         # Becareful categorical variables
         if self.lib=="stats":
            pass
         elif self.lib=="sklearn":
            return Model_selection_sklearn(*plist).model_selection_sklearn_master()
         else:
            pass
