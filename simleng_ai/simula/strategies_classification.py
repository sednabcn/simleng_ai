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

from ..output.table import Table_results
from ..output.graphics import Draw_numerical_results, Draw_binary_classification_results

from scipy import stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as skQDA

from collections import OrderedDict, defaultdict

from ..simula.binary_classification_statsmodels import Binary_classification_statsmodels

from ..simula.multi_classification_statsmodels import Multi_classification_statsmodels

from ..simula.one_label_classification_sklearn import One_label_classification_sklearn


class Classification(Data_Generation):
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
                
    def strategies_classification_master(self):
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

         self.nclass =int(self.target["NCLASS"])

         #nov 24/2023
         self.ntarget= np.max(1,int(self.target["NTARGET"]))
         
         if (self.lib=="stats" and self.ntarget==1 and self.nclass==2):
                     return Binary_classification_statsmodels(*plist).\
                         binary_classification_statsmodels_master()
         elif (self.lib=="stats" and and self.ntarget==1 and self.nclass>2):
                     return Multi_classification_statsmodels(*plist).\
                         multi_classification_statsmodels_master()
         #nov 24/2023
         elif (self.lib=="sklearn" and self.ntarget==1 
             return One_label_classification_sklearn(*plist).one_label_classification_sklearn_master()
         elif (self.lib=="sklearn" and self.ntarget>1 and self.nclass==2):
             pass
         elif (self.lib=="sklearn" and self.ntarget>1 and self.nclass>2):
             pass
    
         else:
             pass    
         
                 
