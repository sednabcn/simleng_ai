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
from .simula.strategies_features_selection import Features_selection
from .data_manager.feature_eng import Data_Engineering
from .data_manager.quality import Data_Visualisation, Data_Analytics, Data_Quality

# setting ignore as a parameter and further adding category
warnings.simplefilter(action="ignore", category=(FutureWarning, UserWarning))


def warning_function():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warning_function()  # now warnings will be suppressed

#file_input = "simlengin2308.txt"


class Simleng:
    def __init__(self, file_input, score=None):
        self.score = score
        self.ini = Init(file_input, score)
        self.gen = Data_Generation(file_input, score)
        # self.game=Simleng_strategies()
        self.eng = Data_Engineering()
        # self.vis = Data_Visualisation()
        self.anal = Data_Analytics()
        self.qual = Data_Quality()
        self.strategies = {}
        self.params = {}
        # self.strategies = {}
        self.strategies_client = {}
        self.action={}
        # generation of entry data and process
        (
            _,
            self.dataset,
            _,
            _,
            self.features,
            self.target,
            self.strategies_client,
            self.params,
        ) = self.gen.get_input_info()

    def simulation_strategies(self):
        """strategies management in Simleng"""

        # printing to start running
        dt_starting = datetime.now().strftime("%d/%m/%y %H:%M:%S")
        print("Simleng start running on", dt_starting)

        self.strategies["Features_selection"] = [
            "full_features",
            "features_selection_z_score",
            "K_fold_cross_validation",
            "pca",
            "additional_test",
            "discriminant_methods",
        ]
        self.strategies["Data"] = [
            "split",
            "balance",
            "augment",
            "synthetic",
            "unique_rep",
        ]
        self.strategies["Training"] = ["base", "full", "pre-trained"]
        self.strategies["Stochastic"] = [
            "boostrap",
            "combined",
            "matrices",
            "operational",
            "meshing",
            "variational",
            "app",
        ]
        self.strategies["Solver"] = [
            "optimizer",
            "parameters",
            "convergence",
            "metrics",
        ]

        # Checking strategies pipeline
        if isinstance(self.strategies_client["STRATEGY"],list):
            self.idoc=[]
            self.vis=[]
            for strategy_i,method_i,idoc_i in zip(self.strategies_client["STRATEGY"],self.strategies_client["METHOD"],self.strategies_client["REPORT"]):
                print(strategy_i,':',method_i)
                checking_input_list(method_i,self.strategies[strategy_i])
                self.idoc.append(int(strtobool(idoc_i)))
                self.vis.append(Data_Visualisation(idoc_i))
        else:
            print(self.strategies_client["STRATEGY"])

            try:
                if (
                        self.strategies_client["METHOD"]
                        not in self.strategies[self.strategies_client["STRATEGY"]]
                ):
                    strategies[self.strategies_client["STRATEGY"]].append(
                        self.strategies_client["METHOD"]
                    )
            except:
                print("Checking the input_file: Simlengin.txt")
              
            # flag to make report
            self.idoc = int(strtobool(self.strategies_client["REPORT"]))
            self.vis = Data_Visualisation(self.idoc)

        return self.strategies_master()

   
    # mastering a type of strategies
    def strategies_features_master(self):
        # Here is taken the class with the same name to the startegy
        from .simula.strategies_features_selection import Features_selection

        if self.action["strategy"]="Features_selection":
                pepe = [
                    self.data_train,
                    self.data_test,
                    self.data_dummy_train,
                    self.data_dummy_test,
                    self.action["method"],
                    self.params,
                    self.action["idoc"],
            ]
            return Features_selection(*pepe).strategies_features_selection_master()

        elif self.action["strategy"] == "Features_extraction":
            pass

    def switch(self, mlmethod, goal, nclass, strategy):
        "Match function for python3.8"
        if mlmethod == "SUPERV":
            if goal == "CLASSIFICATION":
                if nclass == "2":
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

                elif nclass > 2:
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

        if goal == "CLASSIFICATION":
            nclass = str(self.target["NCLASS"])
            score = self.target["SCORE"]

        # moved to Simleng entry point
        # idoc = np.where(self.strategies_client["REPORT"] == True, 0, -1)
        if isinstance(self.strategies_client["STRATEGY"],list):
            for strategy_i,proc_i,idoc_i in zip(self.strategies_client["STRATEGY"],self.strategies_client["METHOD"],self.idoc)
                  strategy = str(strategy_i)
                  self.action.update{"strategy":strategy}
                  proc = str(proc_i)
                  self.action.update{"method":proc}
                  self.action.update{"idoc":idoc_i}
                  self.switch(mlmethod, goal, nclass, strategy)
                  self.action={}
        else:
                  strategy = str(self.strategies_client["STRATEGY"])
                  self.action.update{"strategy":strategy}
                  proc = str(self.strategies_client["METHOD"])
                  self.action.update{"method":proc}
                  self.action.update{"idoc":self.idoc}
                  self.switch(mlmethod, goal, nclass, strategy)
                  self.action={}

        elapsed = timeit.default_timer() - t0
        x, y, z = str(timedelta(seconds=elapsed)).split(":")

        return print("Simleng runs in {}H:{}M:{}S".format(x, y, z))


# MACROSIN=Simleng(file_input,score=0.90).ini.get_macros()

# print(dict(MACROSIN.items()))

# Simleng(file_input,score=0.90).simulation_strategies()
#Simleng(file_input, score=0.90).simulation_strategies()
