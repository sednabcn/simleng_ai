# ==["gen_col_data","mapping_zero_one","get_pandas_from_groupby","max_in_col_matrix","get_max_from_multi_index","pandas_to_array","diif_series","f_index","g_index","col_data","missing_data","col_data_classification"]


def col_data_classification(X,cat_cols,drop_col=0,value=5):
        """Classification of X matrix"""
        X=X.drop(X.iloc[:,[drop_col]],axis=1)
        cat_cols_org=X.columns[cat_cols]
        cat_cols= [col for col in X.columns if X[col].dtype=="object"]
        if cat_cols == cat_cols_org:
                pass
        else:
            cat_cols=cat_cols_org    
        ord_cols= [col for col in X.columns if min(X[col]) in {0,1} and max(X[col])==value]
        cont_cols=[col for col in X.columns if col not in ord_cols + cat_cols]
        columns=X.columns
        # retrieve numpy array
        Xv=X.values
        #dfset = X.values
        # split into input (X) and output (y) variables
        #X = dfset[:, :-1]
        #y = dfset[:,-1]
        return Xv,columns,cat_cols[:-1],ord_cols,cont_cols

def missing_data(X,value=None):
    # for only one column
    clm=list(X.columns[X.isna().sum()>0])
    value=X[clm].median()
    return X.replace(np.nan,value,inplace=True)

def col_data(X, arg):
    """X is a pandas reader file"""
    return X[X.columns[arg]]


def gen_col_data(X, n):
    import pandas as pd

    U = pd.DataFrame(index=X.index, columns=range(1))
    for ii in X.columns:
        if X.columns[n] == ii:
            # Becareful only numbers
            U[:] = X[ii].astype(float)
    return U


def mapping_zero_one(X):
    """Checking if X matrix is ok!"""
    import pandas as pd

    eps = 1e-15
    f = lambda x: (x - x.min()) / (x.max() - x.min() + eps)
    U = pd.DataFrame(X, columns=X.columns)
    for ii in X.columns:
        U[ii] = f(X[ii]).astype(float)
    return U


def get_pandas_from_groupby(x, y, k):
    import numpy as np
    import pandas as pd

    "k-number of groups"

    xy = np.hstack([x, y])
    xy = pd.DataFrame(xy)
    rr = xy.groupby(xy.columns[-1])
    rr = list(rr)

    return [np.array(rr[ii][1].iloc[:, 0]) for ii in range(k)]


def max_in_col_matrix(X, index_col1, index_col2, max_ref):
    """To identify a location of max in a column of matrix"""
    """
         | X -pandas DataFrame
         | index_col1,index_col2-index of columns
         | max_ref-level max. of reference
         """
    import numpy as np

    if len(X[index_col1]) > 0:
        imax1 = max(X[index_col1])
        sa = X.index[X.loc[:, index_col1] == imax1]
        aa = np.where((X.loc[:, index_col1] == imax1) == True, 1, 0)
        Na = max(np.cumsum(aa))

    if len(X[index_col2]) > 0:
        imax2 = max(X[index_col2])
        sb = X.index[X.loc[:, index_col2] == imax2]

    name_list = []
    if Na == 1:
        name = sa[0]
        name_list.append(name)

    else:
        count = 0
        for ii in sa:
            if ii in sb:
                count += 1
                name = ii
                name_list.append(name)
    if len(name_list) == 0:
        name = X.index[sa[-1]]  # more number of predictors
        name_list.append(name)
    max_ref = max(imax1, max_ref)
    return name, name_list, imax1, imax2, max_ref


def get_max_from_multi_index(x, par, index_col):
    """Get an ordered list from index_colum..."""
    """
        |x-list of multi-index [x1, x2, x3,x4...]"
        |par-divisor of mod
        |index-colum index =0,1,2,....
        """
    import pandas as pd

    Idata = pd.DataFrame(index=range(len(x)), columns=range(2))
    n = 0
    for ii in x:
        Idata.loc[n, 0], Idata.loc[n, 1] = divmod(ii, par)
        n += 1
    LOC = Idata.loc[:, index_col].argsort()
    iloc = LOC.tolist()[-1]
    return Idata.loc[iloc, :]


def pandas_to_array(x, kind):
    import numpy as np

    if kind == "r":
        xx = np.ravel(x)
    elif kind == "c":
        xx = np.ravel(transpose(x))
    else:
        print("Error : kind= r (rows) or c (columns)")
    return xx


def diff_series(x, y):
    "Get diff between two series"
    from ..resources.pandas import f_index, g_index
    import pandas as pd

    x = pd.Series(x)
    y = pd.Series(y)

    index_x = f_index(x)
    index_y = g_index(y, index_x)
    index_diff = {}
    for name in index_x.keys():
        if name not in index_y.keys():
            index_diff[name] = index_x[name]
    return pd.Series(index_diff)


def f_index(x):
    """Get index given a prototype..."""
    index = {}
    ii = 0
    for name in x:
        index[name] = ii
        ii += 1
    return index


def g_index(y, x):
    # get y_index given x_index
    index_y = {}
    for name in y:
        index_y[name] = x[name]
    return index_y
