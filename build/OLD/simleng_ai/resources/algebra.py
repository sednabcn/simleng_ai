# ==['precision_matrix","float_with_format","float_list_with_format","solver_eqs_system_2_3","array_mapping_zero_one","add_data","prod_data","mean_rows_data","dot_vector_rows_data"]


def precision_matrix():
    import numpy as np

    Cov = np.array(
        [
            [1.0, 0.9, 0.9, 0.0, 0.0, 0.0],
            [0.9, 1.0, 0.9, 0.0, 0.0, 0.0],
            [0.9, 0.9, 1.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.9, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    print("# Precision matrix:")
    Prec = np.linalg.inv(Cov)
    print(Prec.round(2))
    print("# Partial correlations:")
    Pcor = np.zeros(Prec.shape)
    Pcor[::] = np.NaN
    for i, j in zip(*np.triu_indices_from(Prec, 1)):
        Pcor[i, j] = -Prec[i, j] / np.sqrt(Prec[i, i] * Prec[j, j])
    print(Pcor.round(2))


def float_with_format(x, nplaces):
    return float(
        "".join(seq for seq in ["{0:.", str(nplaces), "f}"]).format(round(x, nplaces))
    )


def float_list_with_format(X, nplaces):
    xx = []
    for ff in X:
        xx.append(float_with_format(ff, nplaces))
    return xx


def solver_eqs_system_2_3(coef_x_y, z_value):
    """Get the solution of equation 05=F(theta*X)"""
    """
        |DataFrame coef_x_y: params= models_params_table contains model.params of each fiiting model
        |z_value: F^(-1)(0.5)
        """
    import numpy as np

    # f=lambda alpha,x,y,z: np.where(z!=0,[(alpha-x)/z,-y/z],np.where(y!=0,(alpha-x)/y,0))
    def f(alpha, x, y, z):
        if np.abs(z) > 1e-10:
            return [(alpha - x) / z, -y / z]
        else:
            return [(alpha - x) / y, -1]
        # return np.piecewise(z,[np.abs(z)>1e-10, np.abs(z)< 1e-10 ],[lambda alpha,x,y,z:[(alpha-x)/z,-y/z],\
        #           lambda alpha,x,y,z: [(alpha-x)/y,-1]])

    solution = []
    for alpha, beta in zip(z_value, coef_x_y.T):
        solution.append(f(alpha, beta[0], beta[1], beta[2]))
    return np.array(solution)


def array_mapping_zero_one(X):
    """Only for an array no pandas"""
    import numpy as np

    X = np.array(X)
    eps = 1e-15
    xmin = X.min()
    xmax = X.max()
    f = lambda x: (x - xmin) / (xmax - xmin + eps)
    try:
        n, m = X.shape
        if m > n:
            X = np.transpose(X)
    except:
        n = len(X)
    U = np.array(X)
    for ii in range(n):
        U[ii] = f(X[ii]).astype(float)
    return U


def add_data(x):
    import numpy as np

    n, m = x.shape
    x = np.array(x)
    sum_by_rows = [x[ii, :].sum() for ii in range(n)]
    return np.asarray(sum_by_rows).astype(float)


def prod_data(x):
    import numpy as np

    n, m = x.shape
    x = np.array(x)
    prod = np.array(np.ones((1, n)))
    for ii in range(m):
        prod = prod * np.array(x[:, ii])
    return prod.astype(float)


def mean_rows_data(x):
    import numpy as np

    n, m = x.shape
    x = np.array(x)
    mean_rows = [x[ii, :].mean() for ii in range(n)]
    mean_rows = np.asarray(mean_rows)
    return mean_rows.astype(float)


def dot_vector_rows_data(b, x):
    import numpy as np

    n, m = x.shape
    x = np.array(x)
    if b.size == m:
        return np.dot(b, np.transpose(x)).astype(float)
    else:
        print("Error in dimensions")
