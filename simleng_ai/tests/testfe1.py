#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:03:55 2023

@author: delta
"""

import pandas as pd
from ..data_manager.feature_eng import Correlation

filename='/home/delta/Downloads/DATASETS/pimas/pima-indians-diabetes.csv'
df=pd.read_csv(filename)
columns=df.columns[:-1]
"""
runfile('/home/delta/Downloads/simleng_ai/testfe1.py', wdir='/home/delta/Downloads/simleng_ai')

columns
Out[3]: Index(['npreg', 'glu', 'bp', 'skin', 'ins', 'bmi', 'ped', 'age'], dtype='object')

df.head()
Out[4]: 
   npreg  glu  bp  skin  ins   bmi    ped  age  type
0      6  148  72    35    0  33.6  0.627   50     1
1      1   85  66    29    0  26.6  0.351   31     0
2      8  183  64     0    0  23.3  0.672   32     1
3      1   89  66    23   94  28.1  0.167   21     0
4      0  137  40    35  168  43.1  2.288   33     1
"""
cr=Correlation(df,columns)
#Test correlation_training
cr.correlation_training()

"""
runfile('/home/delta/Downloads/simleng_ai/testfe1.py', wdir='/home/delta/Downloads/simleng_ai')
Reloaded modules: data_abstract, resources, resources.scrapping, resources.io, init, data_manager, resources.output, data_manager.feature_eng


            Correlation Matrix on Training Data             
╒═══════╤═════════╤═══════╤═══════╤════════╤════════╤═══════╤════════╤════════╤════════╕
│       │   npreg │   glu │    bp │   skin │    ins │   bmi │    ped │    age │   type │
╞═══════╪═════════╪═══════╪═══════╪════════╪════════╪═══════╪════════╪════════╪════════╡
│ npreg │   1.000 │ 0.129 │ 0.141 │ -0.082 │ -0.074 │ 0.018 │ -0.034 │  0.544 │  0.222 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│  glu  │   0.129 │ 1.000 │ 0.153 │  0.057 │  0.331 │ 0.221 │  0.137 │  0.264 │  0.467 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│  bp   │   0.141 │ 0.153 │ 1.000 │  0.207 │  0.089 │ 0.282 │  0.041 │  0.240 │  0.065 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│ skin  │  -0.082 │ 0.057 │ 0.207 │  1.000 │  0.437 │ 0.393 │  0.184 │ -0.114 │  0.075 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│  ins  │  -0.074 │ 0.331 │ 0.089 │  0.437 │  1.000 │ 0.198 │  0.185 │ -0.042 │  0.131 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│  bmi  │   0.018 │ 0.221 │ 0.282 │  0.393 │  0.198 │ 1.000 │  0.141 │  0.036 │  0.293 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│  ped  │  -0.034 │ 0.137 │ 0.041 │  0.184 │  0.185 │ 0.141 │  1.000 │  0.034 │  0.174 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│  age  │   0.544 │ 0.264 │ 0.240 │ -0.114 │ -0.042 │ 0.036 │  0.034 │  1.000 │  0.238 │
├───────┼─────────┼───────┼───────┼────────┼────────┼───────┼────────┼────────┼────────┤
│ type  │   0.222 │ 0.467 │ 0.065 │  0.075 │  0.131 │ 0.293 │  0.174 │  0.238 │  1.000 │
╘═══════╧═════════╧═══════╧═══════╧════════╧════════╧═══════╧════════╧════════╧════════╛

"""
cr.correlation_level()

"""
Treshold value , correlation=0.90
The predictors are slightly correlationed
"""
from ..data_manager.feature_eng import PCA
pc=PCA(df,5).pca_draw_major_minor_factors('None')

pq=PCA(df,5).pca_show_table()

"""
              Eigenvalues in descending order               
╒════╤══════════════╤═══════════╤═══════════════╕
│    │  Predictors  │   Eig.Abs │   cum_var_exp │
╞════╪══════════════╪═══════════╪═══════════════╡
│  0 │    npreg     │   1806.72 │         33.73 │
├────┼──────────────┼───────────┼───────────────┤
│  1 │     glu      │   1362.67 │         59.18 │
├────┼──────────────┼───────────┼───────────────┤
│  2 │      bp      │    860.33 │         75.24 │
├────┼──────────────┼───────────┼───────────────┤
│  3 │     skin     │    677.34 │         87.89 │
├────┼──────────────┼───────────┼───────────────┤
│  4 │     ins      │    648.67 │        100.00 │
╘════╧══════════════╧═══════════╧═══════════════╛
"""

pr=PCA(df,5).pca_draw_by_components()

pn=PCA(df,8).pca_transformation()
"""
print(pn)
         npreg        glu         bp  ...       bmi       ped       age
0   -75.714249 -35.954944  -7.260683  ...  3.457417 -0.695208  0.374544
1   -82.358466  28.909559  -5.496649  ...  5.591914 -2.572576 -0.039284
2   -74.630229 -67.909633  19.461753  ...  7.139400  4.286099  0.358068
3    11.077206  34.900175  -0.053004  ...  2.583990 -0.810101 -0.082633
4    89.744156  -2.751263  25.213059  ... -9.491475 -3.619619  0.802747
..         ...        ...        ...  ...       ...       ...       ...
763  99.237653  25.083009 -19.534828  ...  3.623291  1.332810 -0.516594
764 -78.641427  -7.685767  -4.137339  ... -2.514074 -0.942007 -0.401160
765  32.112987   3.379222  -1.587972  ...  6.242478  1.720432 -0.316662
766 -80.214095 -14.190595  12.351422  ... -2.849050 -5.116701  0.599530
767 -81.308347  21.623042  -8.152774  ...  3.175052 -1.282044 -0.115769

[768 rows x 8 columns]
"""
from ..data_manager.feature_eng import SVD
SVD.svd()

from ..data_manager.feature_eng import Best_features_filter
columns=df.columns[:-1]
Best_features_filter(df[columns],columns,10).variance_influence_factors()
"""
Variance Influence Factors
         VIF Factor
-----  ------------
 bmi         18.409
 glu         16.725
 bp          14.620
 age         13.493
skin          4.009
npreg         3.276
 ped          3.196
 ins          2.064
The following features are collinears: bmi,glu,bp,age
"""
pt=Best_features_filter(df[columns],columns,0.95).show_best_pca_predictors()
"""
['npreg', 'glu', 'bp', 'skin', 'ins', 'ped']
"""
from  ..data_manager.feature_eng import Best_features_wrap
