# ================================================================================
# CHECKING HERE TESTING MODEL AND BACK SCORES OF THEM
# print("== Logistic Ridge (L2 penalty) ==")
# model = lm.LogisticRegressionCV(class_weight='balanced', scoring=balanced_acc, n_jobs=-1)
# Let sklearn select a list of alphas with default LOO-CV (N=K)
# scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
# print("Test ACC:%.2f" % scores.mean())
# ===============================================================================

'''
    def cross_validation_classification():
        
        X, y = datasets.make_classification(n_samples=100, n_features=100,
                                            n_informative=10, random_state=42)

        model = lm.LogisticRegression(C=1)
        cv = StratifiedKFold(n_splits=5,)
        y_test_pred = np.zeros(len(y))
        y_train_pred = np.zeros(len(y))
        for train, test in cv.split(X,y):
            X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
            model.fit(X_train, y_train)
            y_test_pred[test] = model.predict(X_test)
            y_train_pred[train] = model.predict(X_train)
    
            recall_test = metrics.recall_score(y, y_test_pred, average=None)
            recall_train = metrics.recall_score(y, y_train_pred, average=None)
            acc_test = metrics.accuracy_score(y, y_test_pred)
            print("Train SPC:%.2f; SEN:%.2f" % tuple(recall_train))
            print("Test SPC:%.2f; SEN:%.2f" % tuple(recall_test))
            print("Test ACC:%.2f" % acc_test)

        #Scikit-learn provides user-friendly function to perform CV:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
        scores.mean()
        # provide CV and score
    def balanced_acc(estimator, X, y):
            
            Balanced acuracy scorer
            
            return metrics.recall_score(y, estimator.predict(X), average=None).mean()
        #CHECKING
        #scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring=balanced_acc)
        #print("Test ACC:%.2f" % scores.mean())
        
    def cross_validation_regression():
        X, y = datasets.make_regression(n_samples=100, n_features=100,
                                        n_informative=10, random_state=42)
        model = lm.Ridge(alpha=10)
        cv = KFold(n_splits=5, random_state=42,shuffle=False)
        y_test_pred = np.zeros(len(y))
        y_train_pred = np.zeros(len(y))

        for train, test in cv.split(X):
               X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
               model.fit(X_train, y_train)
               y_test_pred[test] = model.predict(X_test)
               y_train_pred[train] = model.predict(X_train)
               print("Train r2:%.2f" % metrics.r2_score(y, y_train_pred))
               print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred))
               #Scikit-learn provides user-friendly function to perform CV:
               scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
               print("Test r2:%.2f" % scores.mean())
               # provide a cv
               scores = cross_val_score(estimator=model, X=X, y=y, cv=cv)
               print("Test r2:%.2f" % scores.mean())

    def cross_validation_regression():
        # Dataset
        X, y, coef = datasets.make_regression(n_samples=50, n_features=100,
                                              noise=10,
        n_informative=2, random_state=42, coef=True)
        print("== Ridge (L2 penalty) ==")
        model = lm.RidgeCV()
        # Let sklearn select a list of alphas with default LOO-CV
        scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
        print("Test r2:%.2f" % scores.mean())
        print("== Lasso (L1 penalty) ==")
        model = lm.LassoCV(n_jobs=-1)
        # Let sklearn select a list of alphas with default 3CV
        scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
        print("Test r2:%.2f" % scores.mean())
        print("== ElasticNet (L1 penalty) ==")
        model = lm.ElasticNetCV(l1_ratio=[.1, .5, .9], n_jobs=-1)
        # Let sklearn select a list of alphas with default 3CV
        scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
        print("Test r2:%.2f" % scores.mean())

    def cross_validation_model_selection():
        # Dataset
        noise_sd = 10
        X, y, coef = datasets.make_regression(n_samples=50,\
        n_features=100, noise=noise_sd, n_informative=2, random_state=42, coef=True)

        # Use this to tune the noise parameter such that snr < 5
        print("SNR:", np.std(np.dot(X, coef)) / noise_sd)

        # param grid over alpha & l1_ratio
        param_grid = {'alpha': 10. ** np.arange(-3, 3), 'l1_ratio':[.1, .5, .9]}

        # Warp
        model = GridSearchCV(lm.ElasticNet(max_iter=10000), param_grid, cv=5)
        # 1) Biased usage: fit on all data, ommit outer CV loop
        model.fit(X, y)

        print("Train r2:%.2f" % metrics.r2_score(y, model.predict(X)))
        print(model.best_params_)

        # 2) User made outer CV, useful to extract specific information
        cv = KFold(n_splits=5, random_state=42)
        y_test_pred = np.zeros(len(y))
        y_train_pred = np.zeros(len(y))
        alphas = list()
        for train, test in cv.split(X):
           X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
           model.fit(X_train, y_train)
           y_test_pred[test] = model.predict(X_test)
           y_train_pred[train] = model.predict(X_train)
           alphas.append(model.best_params_)
    
           print("Train r2:%.2f" % metrics.r2_score(y, y_train_pred))
           print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred))
           print("Selected alphas:", alphas)

           # 3.) user-friendly sklearn for outer CV

           scores = cross_val_score(estimator=model, X=X, y=y, cv=cv)
           print("Test r2:%.2f" % scores.mean())

    def cross_model_selection_bic():
        
        iris = datasets.load_iris()

        X = iris.data
        y_iris = iris.target
        bic = list()
        #print(X)
        ks = np.arange(1, 10)
        for k in ks:
            gmm = GaussianMixture(n_components=k, covariance_type='full')
            gmm.fit(X)
            bic.append(gmm.bic(X))
            k_chosen = ks[np.argmin(bic)]
            plt.plot(ks, bic)
            plt.xlabel("k")
            plt.ylabel("BIC")
            print("Choose k=", k_chosen)
            plt.show()

    def fit_on_increasing_size(model):
        n_samples = 100
        n_features_ = np.arange(10, 800, 20)
        r2_train, r2_test, snr = [], [], []
        for n_features in n_features_:
            # Sample the dataset (* 2 nb of samples)
            n_features_info = int(n_features/10)
            np.random.seed(42) # Make reproducible
            X = np.random.randn(n_samples * 2, n_features)
            beta = np.zeros(n_features)
            beta[:n_features_info] = 1
            Xbeta = np.dot(X, beta)
            eps = np.random.randn(n_samples * 2)
            y = Xbeta + eps
            # Split the dataset into train and test sample
            Xtrain, Xtest = X[:n_samples, :], X[n_samples:, :]
            ytrain, ytest = y[:n_samples], y[n_samples:]
            # fit/predict
            lr = model.fit(Xtrain, ytrain)
            y_pred_train = lr.predict(Xtrain)
            y_pred_test = lr.predict(Xtest)
            snr.append(Xbeta.std() / eps.std())
            r2_train.append(metrics.r2_score(ytrain, y_pred_train))
            r2_test.append(metrics.r2_score(ytest, y_pred_test))
            return n_features_, np.array(r2_train), np.array(r2_test), np.array(snr)

    def plot_r2_snr(n_features_, r2_train, r2_test, xvline, snr, ax):
        """
        #Two scales plot. Left y-axis: train test r-squared. Right y-axis SNR.
        """
        ax.plot(n_features_, r2_train, label="Train r-squared", linewidth=2)
        ax.plot(n_features_, r2_test, label="Test r-squared", linewidth=2)
        ax.axvline(x=xvline, linewidth=2, color='k', ls='--')
        ax.axhline(y=0, linewidth=1, color='k', ls='--')
        ax.set_ylim(-0.2, 1.1)
        ax.set_xlabel("Number of input features")
        ax.set_ylabel("r-squared")
        ax.legend(loc='best')
        ax.set_title("Prediction perf.")
        ax_right = ax.twinx()
        ax_right.plot(n_features_, snr, 'r-', label="SNR", linewidth=1)
        ax_right.set_ylabel("SNR", color='r')

        for tl in ax_right.get_yticklabels():
            tl.set_color('r')
            # Model = linear regression
            mod = lm.LinearRegression()
            # Fit models on dataset
            n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=mod)
            argmax = n_features[np.argmax(r2_test)]
        # plot
            fig, axis = plt.subplots(1, 2, figsize=(9, 3))
        # Left pane: all features
            plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0])
        # Right pane: Zoom on 100 first features
            plot_r2_snr(n_features[n_features <= 100],
            r2_train[n_features <= 100], r2_test[n_features <= 100],
            argmax,
            snr[n_features <= 100],axis[1])
            plt.tight_layout()
            plt.show()
    
'''
