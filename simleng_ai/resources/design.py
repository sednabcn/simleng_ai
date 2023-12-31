#=["macro_strategies","update_macros_strategies_client",]

def macro_strategies(macros,**kwargs):
    dict0={}
    
    if isinstance(macros,list):
        for key,values in kwargs.items():
           if key in macros:
              if key=="Features_selection":
                 dict0[key]= [
                      "full_features",
                      "features_z_score",
                      "K_fold_cross_validation",
                      "with_pca",
                      "KBest",
                      "f_classif",
                      "f_regression",
                      "VarianceThreshold",
                      "RFE",
                      "RFECV",
                      "mutual_info_classif",
                      "mutual_info_regression",
                      "FromModel",
                      "sequential_feature",
                      "chi2",
                        ]
              elif key=="Features_extraction":
                  dict0[key]=[
                      "DictVectorizer",
                      "OneHotEncoder",
                      
                 ]
              elif key=="Model_selection":
                 # MODIFY Nov 24,2023 
                 dict0[key]= [
                      "K_fold_cross_validation",
                      "with_pca",
                      "cross_val_score",
                      "StratifiedKFold",
                      "GridSearchCV",
                      "learning_curve",
                      "validation_curve",
               
                 ]   
              elif key=="Classification":
                 dict0[key]= [
                      "full_features",
                      "features_z_score",
                      "K_fold_cross_validation",
                      "with_pca",
                      "additional_test",
                      "discriminant_methods",
                      "binary_classifiers",
                      "multiple_classifiers",
                 ]

              elif key=="Data":
                   dict0[key]= ["split",
                              "balance",
                              "augment",
                              "synthetic",
                              "unique_rep",
                   ]
                   
              elif key=="Training":
                   dict0[key]= ["base", "full", "pre-trained",]
                 
              elif key=="Stochastic":
                   dict0[key]=  [
                       "boostrap",
                       "combined",
                       "matrices",
                       "operational",
                       "meshing",
                       "variational",
                       "app",
                   ]
                   
              elif key=="Solver":
                   dict0[key]= [
                       "optimizer",
                       "parameters",
                       "convergence",
                       "metrics",
                   ]



                  
           else:
              macros.append(key)
              dict0[key]=[]
           
    return macros,dict0

def update_macros_strategies_client(dataset_name,macr_root,key_client,client,new_macr,default_value):
        """get idoc,lib,vis"""
        """
        dataset_name: name of dataset (str)
        macr_root : strategies
        client : strategies_client
        key_client : "STRATEGY"
        new_macr: new library in strategies_client
        default_value: known member of a new_macr
        """
        from ..data_manager.quality import Data_Visualisation
        from distutils.util import strtobool
        from .io import checking_input_list
        
        

        # modification to include the new_macr column in STRATEGY MACR
        try:
            if ((new_macr not in client.keys()) and \
            (isinstance(client[key_client],list))) :
                
                    client.update({new_macr:[]})

                    for nn in range(len(client[key_client])):
                       client[new_macr].append(default_value)
                       
            elif ((new_macr not in client.keys()) and \
                  ~(isinstance(client[key_client],list))) :

                   client.update({new_macr:default_value})
        
            elif len(client[new_macr])==len(client[key_client]):
                  for nn in range(len(client[new_macr])):
                        if client[new_macr][nn] not in [None,"__","--"," "]:
                                  pass
                        else:
                           client[new_macr][nn]=default_value
            elif ~(isinstance(client[key_client],list)):
                  if client[new_macr] not in [None,"__","--"," "]:
                                  pass
                  else:
                    client[new_macr]=default_value
                                  
            else:
                pass
        except:
                print("Checking the input_file: Simlengin.txt")

        idoc=[]
        vis= []
        new_=[]
            
        if isinstance(client[key_client],list) :
                      
            for strategy_i,method_i,new_i,idoc_i in \
                zip(client[key_client],client["METHOD"],\
                    client[new_macr],client["REPORT"]):
                
                #print(strategy_i,':',method_i)
                checking_input_list(method_i,macr_root[strategy_i])
                new_.append(new_i)
                idoc.append(int(strtobool(idoc_i)))
                vis.append(Data_Visualisation(idoc_i,dataset_name))
        else:
            
            #print(client[key_client])

            try:
                if (client["METHOD"] not in macr_root[client[key_client]]):
                    macr_root[client[key_client]].append(client["METHOD"])
            except:
                print("Checking the input_file: Simlengin.txt")
              
            # flag to make report
            idoc = int(strtobool(client["REPORT"]))
            vis = Data_Visualisation(idoc,dataset_name)
            new_=client[new_macr]
            

        return idoc,vis,new_
