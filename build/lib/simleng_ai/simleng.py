#!/usr/bin/env python3
# three strategy parameters
# first: varg:data, training,stats methods and solver strategies
# second: var: pick a strategy in each varg
# third: do/not do report
# =================================================
# Here script to input the (file_input)
# Here script to update the db: datasets
# Here script to make a cycle till got the score with multi-task
import warnings
import distutils
import timeit
import numpy as np
from datetime import datetime, timedelta
from distutils.util import strtobool
from .ini.init import Init
from .data_manager.generation import Data_Generation
from .data_manager.feature_eng import Data_Engineering
from .data_manager.quality import Data_Visualisation, Data_Analytics, Data_Quality

from .resources.io import checking_input_list
from .resources.design import macro_strategies,update_macros_strategies_client

from .featureseng.strategies_features_selection import Features_selection
from .simula.strategies_classification import Classification

# setting ignore as a parameter and further adding category
warnings.simplefilter(action="ignore", category=(FutureWarning, UserWarning))


def warning_function():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warning_function()  # now warnings will be suppressed

file_input ='simlengin21090917.txt'


class Simleng:
    def __init__(self, file_input, score=None):
        self.score = score
        self.ini = Init(file_input, score)
        self.gen = Data_Generation(file_input, score)
        # self.game=Simleng_strategies()
        self.eng = Data_Engineering()
        # self.vis = Data_Visualisation()
     
        self.qual = Data_Quality()
        self.strategies = {}
        self.params = {}
        # self.strategies = {}
        self.strategies_client = {}
        self.action={}
        # generation of entry data and process
        (
            _,
            self.dataset, # no list yet
            _,
            _,
            self.features,
            self.target,
            self.strategies_client,
            self.params,
        ) = self.gen.get_input_info()
      
        self.dataset_name=self.dataset["DATASET"]
        
        self.anal = Data_Analytics(self.dataset_name)

            
    def simulation_strategies(self):
        """strategies management in Simleng"""
        __strategies__=["Features_selection","Classification","Data","Training",\
                        "Stochastic","Solver"]
        # printing to start running
        dt_starting = datetime.now().strftime("%d/%m/%y %H:%M:%S")

        print("Simleng start running on", dt_starting)
        
        for key in __strategies__:
            self.strategies[key]=[]
        __strategies__,self.strategies=\
            macro_strategies(__strategies__,**self.strategies)

        
        # Checking strategies pipeline
        self.idoc,self.vis,self.lib=update_macros_strategies_client(self.dataset_name,self.strategies,"STRATEGY",self.strategies_client,"LIBRARY","stats")
        
        _,_,self.make_task=update_macros_strategies_client(self.dataset_name,\
            self.strategies,"STRATEGY",self.strategies_client,"MAKETASK",True)

    
        
        return self.strategies_master()

   
    # mastering a type of strategies
    def strategies_features_master(self):
        # Here is taken the class with the same name to the startegy
        #from .featureseng.strategies_features_selection import Features_selection 
        # REQUIRE IMPROVEMENT TAKING A SWITCH METHOD
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

        if self.action["strategy"]=="Features_selection":
            print(self.action["strategy"])
            return Features_selection(*plist).\
                             strategies_features_selection_master()    

        elif self.action["strategy"] == "Classification":
            print(self.action["strategy"])
            return Classification(*plist).\
                                strategies_classification_master()
        else:
             pass

    def switch(self, mlmethod, goal, strategy):
        "Match function for python3.8"
        if mlmethod == "SUPERV":
            if goal == "CLASSIFICATION":
                
                    (
                        self.data_train,
                        self.data_test,
                        _,
                        self.X_train,
                        self.X_test,
                        self.y_train,
                        self.y_test,
                        _,
                        _,
                        _,
                        _,
                    ) = self.gen.data_generation_functions()
                    
                    (
                        self.data_dummy_train,
                        self.data_dummy_test,
                    ) = self.gen.data_generation_target()

                    if strategy == "Features_selection":
                        return self.strategies_features_master()

                    elif strategy == "Features_extraction":
                        pass

                    elif strategy =="Classification":
                        return self.strategies_features_master()
                    
                    elif strategy == "Data":
                        pass
                    elif strategy == "Training":
                        pass
                    elif strategy == "Stochastic":
                        pass
                    elif strategy == "Solver":
                        pass
                    else:
                        pass

            elif goal == "REGRESSION":
                pass
            elif goal == "DIMENSION REDUCTION":
                pass
            else:
                pass
        elif mlmethod == "UNSUPERV":
            pass
        elif mlmethod == "SEMI-SUPERV":
            pass

        else:
            pass

    def strategies_master(self):
        import numpy as np

        # starting running time of Simleng
        t0 = timeit.default_timer()

        # parameters to characteize the strategy

        mlmethod = str(self.target["METHOD"])
        kind = str(self.dataset["TYPE"])
        struct = str(self.dataset["STRUCT"])
        goal = str(self.target["GOAL"])
        score = self.target["SCORE"]
        
        # including features selection from other libraries

        solver = str(self.target["SOLVER"])
        
        
        if isinstance(self.strategies_client["STRATEGY"],list):

            for strategy_i,proc_i,lib_i,idoc_i,task_i in zip(self.strategies_client["STRATEGY"],\
                                                      self.strategies_client["METHOD"],\
                                                             self.lib,self.idoc,\
                                                             self.make_task):
                  strategy = str(strategy_i)
                  self.action.update({"strategy":strategy})
                  proc = str(proc_i)
                  self.action.update({"method":proc})
                  lib =str(lib_i)
                  self.action.update({"library":lib})
                  self.action.update({"idoc":idoc_i})
                  
                  self.action.update({"make_task":task_i})
                  self.switch(mlmethod, goal,strategy)
                  self.action={}
        else:
                  strategy = str(self.strategies_client["STRATEGY"])
                  self.action.update({"strategy":strategy})
                  proc = str(self.strategies_client["METHOD"])
                  self.action.update({"method":proc})
                  lib = str(self.target["SOLVER"])
                  self.action.update({"library":lib})
                  self.action.update({"idoc":self.idoc})
                  self.action.update({"make_task":self.make_task})
                  self.switch(mlmethod, goal,strategy)
                  self.action={}

        elapsed = timeit.default_timer() - t0
        x, y, z = str(timedelta(seconds=elapsed)).split(":")

        return print("Simleng runs in {}H:{}M:{}S".format(x, y, z))


#MACROSIN=Simleng(file_input,score=0.90).ini.get_macros()

#print(dict(MACROSIN.items()))

Simleng(file_input,score=0.90).simulation_strategies()
#Simleng(file_input, score=0.90).simulation_strategies()
