# ==["find_subsets_predictors","data_add_constant",data_sorted", "data_dummy_binary_classification","data_0_1_exposure","data_add_exposure","boostrap_datasets","isdummy","isimbalance"]

'''
#def find_subsets_predictors(x,dict_index,shuffle_mode,filter_cond):
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
    from ..resources.sets import order_mode
    
    s_index=dict_index["list_index_col_subset"]
    k_base=dict_index["min_subset_size"]
    df=x
    k_search=dict_index["subset_size_list"]

    
    subsets_all=[]
    match shuffle_mode:
        case "combinations":
              subsets_all.extend(list(combinations(s_index,k_base))) 
              if len(k_search)>0:
                  for k in k_search:
                          if k>k_base:
                             for item in list(combinations(s_index,k)):
                                   subsets_all.append(item)
                             # subsets_all.extend(list(combinations(s_index,k) for k in k_search if k>k_base))
                             
        case "permutations":
              subsets_all.extend(list(permutations(s_index,k_base)))
              if len(k_search)>0:
                  for k in k_search:
                          if k>k_base:
                             for item in list(combinations(s_index,k)):
                                   subsets_all.append(item)
        case "None":
              subsets_all.extend(list(combinations(s_index,k_base))) 
              
    #print(subsets_all)          
    columns=[df.columns[list(item)].tolist() for item in subsets_all]
    # print(columns)

    if filter_cond=="same":
       columns=[item for item in columns if order_mode(df.columns,item)] 
    else:
        pass
    print(type(columns))
    print(type(subsets_all))
    print(columns)
    print("=============================")
    print(subsets_all)
    print("=============================")
    
    return subsets_all,columns
'''


def isdummy(x):
    import numpy as np
    import pandas as pd

    dummy = True
    xx = pd.DataFrame(x)
    ind = xx.values.max()
    return np.where(ind > 1, not (dummy), dummy)


def isimbalance(x):
    import pandas as pd
    import numpy as np

    imbalance = False
    X = pd.DataFrame(x)
    weight = X.groupby(X.columns[-1]).count()[0]
    return np.where(np.all(weight == weight[0]), not (imbalance), imbalance)


def data_add_constant(x):
    import statsmodels.api as sm

    exog = sm.add_constant(x, prepend=True)
    return exog


def data_sorted(x, magnitud, key, reverse):
    "Sorted a list,dict,array,tuple.."
    """
             | x : data
             | magnitud :lambda function to apply to x
             | key: lambda function to define order criteria
             | reverse: bool
             """
    return sorted(enumerate(list(map(magnitud, x))), key=key, reverse=reverse)


def data_dummy_binary_classification(y, key, L1, L2):
    # Mapping 'type' (Yes/No) to (1/0)
    import numpy as np

    f = lambda x: np.where(x == key, L1, L2)
    V = f(y).astype(int)
    V.reshape(len(y), 1)
    return V


def data_0_1_exposure(data):
    """Transformar data to [0,1] by limits"""
    import numpy as np

    f = lambda x: np.where(np.abs(x) > 1, 1, 0)
    return f(data)


def data_add_exposure(data, loc, var):
    """Add perturbation: u= N(0,1) to X_train..."""
    from add_exposure import Add_exposure
    import numpy as np
    import pandas as pd

    sadd = Add_exposure().normal_0_1(loc, var, data.shape[0])
    sadd = np.array(sadd, dtype=float).reshape(data.shape[0], 1)
    zz = np.zeros((data.shape))
    sadd_table = pd.DataFrame(
        zz, index=range(1, data.shape[0] + 1), columns=data.columns
    )
    for ii in range(len(sadd_table.columns)):
        sadd_table[sadd_table.columns[ii]] = sadd
    data += sadd_table
    return data


def find_subsets_predictors(list_index_col_subset, data, shuffle_size, shuffle_mode):
    pass
    return


def boostrap_datasets():
    # Bootstrap loop
    nboot = 100  # !! Should be at least 1000
    scores_names = ["r2"]
    scores_boot = np.zeros((nboot, len(scores_names)))
    coefs_boot = np.zeros((nboot, X.shape[1]))
    orig_all = np.arange(X.shape[0])
    for boot_i in range(nboot):
        boot_tr = np.random.choice(orig_all, size=len(orig_all), replace=True)
        boot_te = np.setdiff1d(orig_all, boot_tr, assume_unique=False)
        Xtr, ytr = X[boot_tr, :], y[boot_tr]
        Xte, yte = X[boot_te, :], y[boot_te]
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte).ravel()
        scores_boot[boot_i, :] = metrics.r2_score(yte, y_pred)
        coefs_boot[boot_i, :] = model.coef_
        # Compute Mean, SE, CI
        scores_boot = pd.DataFrame(scores_boot, columns=scores_names)
        scores_stat = scores_boot.describe(
            percentiles=[0.99, 0.95, 0.5, 0.1, 0.05, 0.01]
        )
        print(
            "r-squared: Mean=%.2f, SE=%.2f, CI=(%.2f %.2f)"
            % tuple(scores_stat.ix[["mean", "std", "5%", "95%"], "r2"])
        )
        coefs_boot = pd.DataFrame(coefs_boot)
        coefs_stat = coefs_boot.describe(percentiles=[0.99, 0.95, 0.5, 0.1, 0.05, 0.01])
        print("Coefficients distribution")
        print(coefs_stat)


class MDS:
    """
    Methods=[mds,non_linear_mds]
    """

    def mds():
        # Pairwise distance between European cities
        try:
            url = "../data/eurodist.csv"
            df = pd.read_csv(url)
        except:
            url = "https://raw.githubusercontent.com/neurospin \
            /pystatsml/master/datasets/eurodist.csv"

        df = pd.read_csv(url)
        print(df.ix[:5, :5])
        city = df["city"]
        D = np.array(df.ix[:, 1:])
        # Distance matrix
        # Arbitrary choice of K=2 components
        skmds = skMDS(
            dissimilarity="precomputed",
            n_components=2,
            random_state=40,
            max_iter=3000,
            eps=1e-9,
        )
        X = skmds.fit_transform(D)
        # Recover coordinates of the cities in Euclidean referential whose orientation is arbitrary:

        Deuclidean = metrics.pairwise.pairwise_distances(X, metric="euclidean")
        print(np.round(Deuclidean[:5, :5]))

        # Plot the results:
        # Plot: apply some rotation and flip
        theta = 80 * np.pi / 180.0
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        Xr = np.dot(X, rot)
        # flip x
        Xr[:, 0] *= -1
        plt.scatter(Xr[:, 0], Xr[:, 1])
        for i in range(len(city)):
            plt.text(Xr[i, 0], Xr[i, 1], city[i])
        plt.axis("equal")
        plt.show()
        # Determining the number of components
        # We must choose K * ∈ {1, . . . , K} the number of required components.
        # Plotting the values of the stress function, obtained using k ≤ N − 1 components.
        # In general, start with 1, . . . K ≤ 4. Choose K * where you can clearly distinguish
        # an elbow in the stress curve.
        # Thus, in the plot below, we choose to retain information accounted for by the first
        # two components, since this is where the elbow is in the stress curve.
        k_range = range(1, min(5, D.shape[0] - 1))
        stress = [
            skMDS(
                dissimilarity="precomputed",
                n_components=k,
                random_state=42,
                max_iter=300,
                eps=1e-9,
            )
            .fit(D)
            .stress_
            for k in k_range
        ]
        print(stress)
        plt.plot(k_range, stress)
        plt.xlabel("k")
        plt.ylabel("stress")
        return plt.show()

    def non_linear_mds():
        #!/usr/bin/env python
        # Nonlinear dimensionality reduction
        # Sources:
        # • Scikit-learn documentation
        # • Wikipedia
        """
        Nonlinear dimensionality reduction or manifold learning cover unsupervised methods
        that attempt to identify low-dimensional manifolds within the original
        P-dimensional space that represent high data density. Then those methods
        provide a mapping from the high-dimensional space to the low-dimensional embedding.
        Isomap
        Isomap is a nonlinear dimensionality reduction method that combines a procedure
        to compute the distance matrix with MDS. The distances calculation is based on
        geodesic distances evaluated on neighborhood graph:
        1. Determine the neighbors of each point. All points in some fixed radius
        or K nearest neighbors.
        2. Construct a neighborhood graph. Each point is connected to other if
        it is a K nearest neighbor. Edge length equal to Euclidean distance.
        3. Compute shortest path between pairwise of points d ij to build the distance matrix D.
        4. Apply MDS on D.
        """

        X, color = datasets.samples_generator.make_s_curve(1000, random_state=42)
        fig = plt.figure(figsize=(10, 5))
        plt.suptitle("Isomap Manifold Learning", fontsize=14)
        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
        ax.view_init(4, -72)
        plt.title('2D "S shape" manifold in 3D')
        Y = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
        ax = fig.add_subplot(122)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.title("Isomap")
        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.axis("tight")
        return plt.show()

    def randomnize_permutation():
        import numpy as np
        import scipy.stats as stats
        import matplotlib.pyplot as plt

        np.random.seed(42)
        x = np.random.normal(loc=10, scale=1, size=100)
        y = x + np.random.normal(loc=-3, scale=3, size=100)  # snr = 1/2
        # Permutation: simulate the null hypothesis
        nperm = 10000
        perms = np.zeros(nperm + 1)
        perms[0] = np.corrcoef(x, y)[0, 1]
        for i in range(1, nperm):
            perms[i] = np.corrcoef(np.random.permutation(x), y)[0, 1]
        # Plot
        # Re-weight to obtain distribution
        weights = np.ones(perms.shape[0]) / perms.shape[0]
        plt.hist(
            [perms[perms >= perms[0]], perms],
            histtype="stepfilled",
            bins=100,
            label=["t>t obs (p-value)", "t<t obs"],
            weights=[weights[perms >= perms[0]], weights],
        )
        plt.xlabel("Statistic distribution under null hypothesis")
        plt.axvline(x=perms[0], color="blue", linewidth=1, label="observed statistic")
        _ = plt.legend(loc="upper left")
        # One-tailed empirical p-value
        pval_perm = np.sum(perms >= perms[0]) / perms.shape[0]
        # Compare with Pearson's correlation test
        _, pval_test = stats.pearsonr(x, y)
        print(
            "Permutation two tailed p-value=%.5f. Pearson test p-value=%.5f"
            % (2 * pval_perm, pval_test)
        )
        return plt.show()
