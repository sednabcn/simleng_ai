#["isop","select_model_name_method","emb_nclass","inv_emb_nclass","find_subsets_predictors",]


def isop(ltasks,task,lmodels,model):
     "bool function to decide the operation is doing"
     from collections import defaultdict 
     isop_dict=defaultdict(dict)
     for x in lmodels:
         for y in ltasks:
             isop_dict[x][y]=False
     isop_dict["OLS"]["LinearRegression"]=True
     isop_dict["MNLogit"]["MultiClassification"]=True
     isop_dict["GLM"]["BinaryClassification"]=True
     isop_dict["GLM"]["LinearRegression"]=True
     isop_dict["Logit"]["BinaryClassification"]=True
     isop_dict["GenLogit"]["BinaryClassification"]=True
     isop_dict["Probit"]["BinaryClassification"]=True
     if task in ltasks and model in lmodels:
         return isop_dict[model][task]
     else:
         print("Error in mmodel selection")

def select_model_name_method(lmodels,model):
        "Dict of model_name used in the program"

        from collections import defaultdict

        model_name=defaultdict()

        model_name["GLM"]="sm.GLM"
        model_name["Logit"]="smd.Logit"
        model_name["GenLogit"]="GenLogit"
        model_name["Probit"]="smd.Probit"
        model_name["OLS"]="sm.OLS"
        model_name["MNLogit"]="smd.MNLogit"
        
        model_method=defaultdict()
        model_method["OLS"]="elastic_net"
        model_method["GLM"]="elastic_net"

        for x in lmodels:
             if x not in ["OLS","GLM"]:
                  model_method[x]="l1"

        return model_name[model],model_method[model]

def emb_nclass(endog_encode,n):
    "Embedding a vector in a matrix id"
    import numpy as np
    import pandas as pd
    m=len(endog_encode)
    pn=np.zeros((m,n),dtype=float)
    for i,value in enumerate(endog_encode):
        pn[i,value]=1.0
    return pd.DataFrame(pn,index=endog_encode.index)

def inv_emb_nclass(vmatrix,n,treshold):
     "Transform results's matrix to id_nclass vector"
     import numpy as np
     import pandas as pd
     m,p=vmatrix.shape
     
     endog_encode=np.empty(m,dtype="object")
     f=lambda x: np.where(x < treshold, 0, 1)
     if p!=n:
        return print("Error in the calculations")
     if p==1:
          endog_encode=np.array(list(map(f,vmatrix)))
          return endog_encode
     elif p > 1:
          for ii in range(p):   
               enc=[j for j,x in enumerate(list(map(f, vmatrix.iloc[:,ii]))) if x==1]
               for jj in enc:
                    endog_encode[jj]=ii
                    
          return pd.Series(endog_encode,index=vmatrix.index) 
     else:
        pass

def find_subsets_predictors(x,*args):
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
        from .sets import order_mode

        df = x
        # input values
        #=========================
        index_columns_base=args[0]
        subset_search=args[1]
        min_shuffle_size=args[2]
        shuffle_mode=args[3]
        filter_cond=args[4]
        #===================
        s_index = [int(num) for num in re.findall(r"\d+", index_columns_base)]

        k_search = [int(num) for num in re.findall(r"\d+", subset_search)]

        k_base = min_shuffle_size

        shuffle_mode = str(shuffle_mode)

        filter_cond = str(filter_cond)

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
