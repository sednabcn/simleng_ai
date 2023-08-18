#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:24:52 2023

@author: agagora
"""


def order_mode(data, tsearch):
    from collections import OrderedDict

    order = OrderedDict()

    list_base = data
    list_search = tsearch
    if not isinstance(data, list):
        list_base = list(data)
    if not isinstance(tsearch, list):
        list_search = []
        for item in tsearch:
            list_item = item
            if not (isinstance(item, list)):
                list_item = list(item)
            list_search.append(list_item)

    for item in list_base:
        order[item] = list_base.index(item)

    f_order = lambda x: order[x]

    order_search = list(map(f_order, list_search))

    if sorted(order_search) != order_search:
        return False
    else:
        return True


def find_subsets_predictors(
    data,
    list_index_col_subset,
    min_subset_size,
    subset_size_list,
    shuffle_mode,
    filter_cond=None,
):
    """Find subsets predictors in features selection methods"""
    """
    Parameters
    ----------

     data: dataframe data
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

    s_index = list_index_col_subset
    k_base = min_subset_size
    df = data
    k_search = subset_size_list

    subsets_all = []
    match shuffle_mode:
        case "combinations":
            subsets_all.extend(list(combinations(s_index, k_base)))
            if len(k_search) > 0:
                for k in k_search:
                    if k > k_base:
                        for item in list(combinations(s_index, k)):
                            subsets_all.append(item)
                        # subsets_all.extend(list(combinations(s_index,k) for k in k_search if k>k_base))

        case "permutations":
            subsets_all.extend(list(permutations(s_index, k_base)))
            if len(k_search) > 0:
                for k in k_search:
                    if k > k_base:
                        for item in list(combinations(s_index, k)):
                            subsets_all.append(item)
        case "None":
            subsets_all.extend(list(combinations(s_index, k_base)))

    # print(subsets_all)
    columns = [df.columns[list(item)].tolist() for item in subsets_all]
    # print(columns)

    if filter_cond == "same":
        columns = [item for item in columns if order_mode(df.columns, item)]
    else:
        pass

    return subsets_all, columns


import pandas as pd

filename = "/home/agagora/Downloads/simleng_ai/datasets/pimas/pima-indians-diabetes.csv"
data = pd.read_csv(filename)
print(data.columns)
print("=======================")
subsets_all, columns_base = find_subsets_predictors(
    data, [0, 1, 4, 5, 6], 3, [3, 4, 5], "None"
)
# print(columns_base)
subsets_all, columns_base = find_subsets_predictors(
    data, [0, 1, 4, 5, 6], 3, [3, 4, 5], "combinations", "same"
)
# subsets_all=find_subsets_predictors(data,[0,1,4,5,6],3,[3,4,5],"combinations","same")
print(columns_base)
# print(subsets_all)
# subsets_all,columns_base=find_subsets_predictors(data,[0,1,4,5,6],3,[3,4,5],"permutations","same")
# print(columns_base)
"""
EXamples
==========
In [27]: runfile('/home/agagora/Downloads/simleng_ai/tests/test_shuffle_subsets.py', wdir='/home/agagora/Downloads/simleng_ai/tests')
       glu  bp  skin  ins   bmi    ped  age  type
npreg                                            
6      148  72    35    0  33.6  0.627   50     1
1       85  66    29    0  26.6  0.351   31     0
8      183  64     0    0  23.3  0.672   32     1
1       89  66    23   94  28.1  0.167   21     0
0      137  40    35  168  43.1  2.288   33     1

shuffle_mode="None"

shuffle_mode="combinations"

"""
