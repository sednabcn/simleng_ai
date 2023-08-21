class Data_Engineering:
    """ Transformation data and features extraction """ 
    #from heatmap import Heatmap, corrplot
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import statsmodels.discrete.discrete_model as smd
    from statsmodels.multivariate.pca import PCA as smPCA
    from sklearn.decomposition import PCA as skPCA
    from resources.output import table
    import matplotlib.pyplot as plt
    from pyensae.graphhelper import corrplot
    from resources.output import table
    from output.graphics import Drawing2d

    def __init__(self):
           pass
       
class Correlation(Data_Engineering):
    """Compute correlation of data and inform about its strength."""

    def __init__(self,data,columns,treshold=0.9):
        self.x=data
        self.columns_subset=columns # selection of columns
        self.corr_treshold=treshold

    def correlation_training(self):
        """Investigation of training data correlation."""
        import pandas as pd
        from resources.output import table
        from pyensae.graphhelper import Corrplot
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        self.x=pd.DataFrame(self.x)

        dfcorr=self.x.corr()
        table(dfcorr,'.3f','fancy_grid','Correlation Matrix on Training Data', 60)
        #c=corrplot.Corrplot(dfcorr)
        #corrplot(dfcorr)
        #figure1
        plt.style.use("ggplot")
        c=Corrplot(self.x)
        c.plot(cmap=('Orange', 'white', 'green'))
        c.plot(method='circle') 
        plt.suptitle("Pair-Correlation Matrix on Training Data")
        plt.show()
        #figure2
        print(dfcorr)
        #plotting correlation heatmap
        dataplot = sns.heatmap(dfcorr, cmap="YlGnBu", annot=True)
        return plt.show(),

    def correlation_level(self):
        """Identify significative correlations"""
        import numpy as np
        import pandas as pd
        self.x=pd.DataFrame(self.x)
        print("\n","Treshold value , correlation=%.2F"%( self.corr_treshold),sep='\n')
        mcorr=self.x.corr()
        bb=np.triu(mcorr)-np.diag(mcorr)
        aa=np.where(bb > self.corr_treshold)
        LL=len(aa[0])
        mcorr_low=[]
        mcorr_high=[]
        if (LL>0):
            for ii in range(LL):
                a=self.x.columns[aa[0][ii]]
                mcorr_high.append(a)
                b=self.x.columns[aa[1][ii]]
                mcorr_high.append(b)

            mcorr_low=[names for names in self.x.columns if names not in mcorr_high]
            return print("The following predictors are highly pair-correlationed:"\
                         + ','.join(str(name) for name in mcorr_high),sep='\n'), print("\n",\
                    "The following predictors are midly pair-correlationed "\
                     +','.join(str(name) for name in mcorr_low),sep='\n')
        else:
             return  print("\n\n","The predictors are slightly correlationed",sep="\n")


class PCA:
    """PCA Analysis."""
    def __init__(self,x,n_components):
         self.x=x
         self.ncomp=n_components

    def pca(self):
         import numpy as np
         from statsmodels.multivariate.pca import PCA as smPCA
         pc=smPCA(self.x,self.ncomp)
         factors=pc.factors.values[:]
         eigvalues=pc.eigenvals
         eigvectors=pc.eigenvecs.values[:]

         # Make a sorted list of (eigenvalue, eigenvector)

         order=[ii for ii,vals in sorted(enumerate(np.abs(eigvalues)), \
                                         key=lambda x:x[1],reverse=True)]

         major_factor=factors[:,order[0]]
         minor_factor=factors[:,order[-1]]


         eigvalues_sorted=[eigvalues[order[ii]] for ii in range(self.ncomp)]
         eigvectors_sorted=[eigvectors[:,order[ii]] for ii in range(self.ncomp) ]


         tot = sum(eigvalues_sorted)
         var_exp = [(i / tot)*100 for i in eigvalues_sorted[:]]
         cum_var_exp = np.cumsum(var_exp)
         return major_factor,minor_factor,factors,eigvalues_sorted,eigvectors_sorted,var_exp,cum_var_exp


    def pca_draw_major_minor_factors(self,target):
        """Draw the major and minor PCA factor."""
        """WORKING TO GENERALIZE"""
        import matplotlib.pyplot as plt
        import numpy as np
        from output.graphics import Drawing2d

        major_factor,minor_factor,_,_,_,explained_variance_,_=PCA(self.x,self.ncomp).pca()

        #major_factor=np.reshape(major_factor,[1,major_factor.shape[0]])
        #minor_factor=np.reshape(minor_factor,[1,minor_factor.shape[0]])
        
        mx=max(explained_variance_)
        mean_=np.mean(explained_variance_)/(100*mx)

        LENGTHS=[explained_variance_[0]/mx,explained_variance_[-1]/mx]

        components=np.array([[major_factor[0],minor_factor[0]],
                             [-minor_factor[0],major_factor[0]]])
        
        plt.figure()
        # color has the same size of X,y
        color=major_factor.shape[0]*['Darkblue'] #['Darkblue','red']
        
        ax=plt.subplot(1,1,1)
        ax.scatter(major_factor,minor_factor,c=color, marker='o',edgecolor='none',\
                   alpha=0.8,s=40)
        for length, vector in zip(LENGTHS,components):
            v = mean_+ 4*np.sqrt(length)*vector
            Drawing2d.draw_vector(v[0],v[1],mean_,mean_)

        plt.xlabel("PCA factor(major axis]")
        plt.ylabel("PCA factor(minor axis]")
        plt.title("Major PCA Factor vs Minor PCA Factor")
        plt.axis("equal")
        return plt.show()

    def pca_show_table(self):
         "Table of eigvalues and eigvectors in reverse order."

         import pandas as pd
         from resources.output import table
         import numpy as np
         _,_,_,eigvalues,eigvectors,_,cum_var_exp=PCA(self.x,self.ncomp).pca()
        
         #eig_data=[(self.x.columns[i],np.abs(eigvalues[i]), \
         #          cum_var_exp[i],eigvectors[i]) for i in range(self.ncomp)]
         eig_data=[(self.x.columns[i],np.abs(eigvalues[i]), \
                    cum_var_exp[i]) for i in range(self.ncomp)]

         eig_data=pd.DataFrame(eig_data,columns=['Predictors','Eig.Abs','cum_var_exp'] )
         #eig_data=pd.DataFrame(eig_data[:,2],columns=['obs_var','Eig.Abs',\
         #                        'cum_var_exp','Eigenvectors'] )
         #return Tools.table(eig_data,'.3f','fancy_grid','Eigenvalues in descending order',60)


         return table(eig_data,'.2f','fancy_grid','Eigenvalues in descending order',60)


    def pca_draw_by_components(self):
         """Visualization PCA by components."""
         import matplotlib.pyplot as plt
         # colors for grahics with matplolib and plotly
         from matplotlib import colors as mcolors
         colors = [ic for ic in mcolors.BASE_COLORS.values() if ic !=(1,1,1)]

         _,_,_,eigvalues,eigvectors, var_exp,cum_var_exp=PCA(self.x,self.ncomp).pca()

         plt.Figure()
         ax=plt.subplot(1,1,1)
         width=0.40
         x_draw=[self.x.columns[i] for i in range(self.ncomp)]
         y_draw=var_exp
         z_draw=cum_var_exp
         plt.bar(x_draw,y_draw,width,color=colors[:7])
         plt.scatter(x_draw,z_draw)
         ax.plot(x_draw,z_draw,'y')
         plt.ylabel('Cumulative explained variance')
         plt.xlabel('Observed variables')
         plt.title("Visualization of PCA by components")
         return plt.show()


    def pca_transformation(self):
        """To apply PCA transformation to Training Data."""
        import pandas as pd
        from sklearn.decomposition import PCA as skPCA
        
        pc=skPCA(n_components=self.ncomp)
        X_pca=pc.fit_transform(self.x)
        X_pca=pd.DataFrame(X_pca,columns=self.x.columns[:self.ncomp])
        return X_pca
class SVD:
    def __init__(self):
        pass
   
    def svd():
        # This procedure was taken from...
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        np.random.seed(42)
        # dataset
        n_samples = 100
        experience = np.random.normal(size=n_samples)
        salary = 1500 + experience + np.random.normal(size=n_samples, scale=.5)
        X = np.column_stack([experience, salary])
        # PCA using SVD
        X -= X.mean(axis=0) # Centering is required
        U, s, Vh = scipy.linalg.svd(X, full_matrices=False)
        # U : Unitary matrix having left singular vectors as columns.
        #Of shape (n_samples,n_samples) or (n_samples,n_comps), depending on
        #full_matrices.
        # s : The singular values, sorted in non-increasing order.
        #Of shape (n_comps,),
        #with n_comps = min(n_samples, n_features).
        # Vh: Unitary matrix having right singular vectors as rows.
        #Of shape (n_features, n_features) or (n_comps, n_features) depending
        # on full_matrices.
        plt.figure(figsize=(9, 3))
        plt.subplot(131)
        plt.scatter(U[:, 0], U[:, 1], s=50)
        plt.axis('equal')
        plt.title("U: Rotated and scaled data")
        plt.subplot(132)
        # Project data
        PC = np.dot(X, Vh.T)
        plt.scatter(PC[:, 0], PC[:, 1], s=50)
        plt.axis('equal')
        plt.title("XV: Rotated data")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.subplot(133)
        plt.scatter(X[:, 0], X[:, 1], s=50)
        for i in range(Vh.shape[0]):
            plt.arrow(x=0, y=0, dx=Vh[i, 0], dy=Vh[i, 1], head_width=0.2,
            head_length=0.2, linewidth=2, fc='r', ec='r')
            plt.text(Vh[i, 0], Vh[i, 1],'v%i' % (i+1), color="r", fontsize=15,
            horizontalalignment='right', verticalalignment='top')
        plt.axis('equal')
        plt.ylim(-4, 4)
        plt.title("X: original data (v1, v2:PC dir.)")
        plt.xlabel("experience")
        plt.ylabel("salary")
        plt.tight_layout()
        plt.show()
        #Principal components analysis (PCA


class Best_features_filter:
    """ To apply criteria to extract features's subsample."""
    """
     1.z_score
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    def __init__ (self,x,columns_subset,vif_treshold):
        self.x=x
        self.columns_subset=columns_subset
        self.vif_treshold=vif_treshold

    def variance_influence_factors(self):
        "Analysis of variance influence factors or colinearity"
        import pandas as pd
        import numpy as np
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from resources.output import table
        
        vif = pd.DataFrame()
        vif["features"] = self.columns_subset
        X_vif=np.asarray(self.x[self.columns_subset])
        vif["VIF Factor"] = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
        vif=np.asarray(vif)
        vif =dict(vif)
        vif_inverse={}
        for key,value in vif.items():
            vif_inverse[value]=key
        keys_ordered=[vif_inverse[name] for name in sorted(vif.values(),reverse=True)]
        vif_ordered=pd.DataFrame(index=keys_ordered,columns=['VIF Factor'])

        for key in keys_ordered:
            vif_ordered.loc[key,:]=vif[key]


        key_colinearity= vif_ordered.index[np.where(vif_ordered.iloc[:,0]>self.vif_treshold)]

        return key_colinearity,table(vif_ordered,'.3f','simple',"Variance Influence Factors",0),\
        print("\n","The following features are collinears: "\
                     + ','.join(key for key in key_colinearity),"\n",sep="\n")


    def show_best_pca_predictors(self):
        
            import numpy as np
            from statsmodels.multivariate.pca import PCA as smPCA
            from data_manager.feature_eng import Best_features_filter

            x=self.x

            ncomp=len(self.columns_subset)
            vif_treshold=self.vif_treshold

            try:
                eigenvalues=smPCA(x,ncomp).eigenvals
            except:
                eigenvalues=smPCA(x,ncomp,method='eig').eigenvals

            order=[ii for ii,vals in sorted(enumerate(np.abs(eigenvalues)), \
                                         key=lambda x:x[1],reverse=True)]

            eigenvalues_PCA_sorted=[eigenvalues[order[ii]] for ii in range(ncomp)]

            features_PCA_sorted=[x.columns[order[ii]] for ii in range(ncomp)]

            key_colinearity,_,_=self.variance_influence_factors()
            #key_colinearity,_,_=Best_features_filter(x,features_PCA_sorted,vif_treshold).variance_influence_factors()

            features_sorted=[]
            eigenvalues_sorted=[]
            for feat,eig in zip(features_PCA_sorted,eigenvalues_PCA_sorted):
                            if feat in key_colinearity[0]:
                                if feat==key_colinearity[0][0]:
                                    features_sorted.append(feat)
                                    eigenvalues_sorted.append(eig)
                            else:
                                 features_sorted.append(feat)
                                 eigenvalues_sorted.append(eig)


            tot = sum(eigenvalues_sorted)
            best_predictors=[]
            cum_var_exp=0.0
            for ii,i in enumerate(eigenvalues_sorted[:]):
                var_exp=(i/tot)
                cum_var_exp+=var_exp
                if cum_var_exp < vif_treshold:
                        best_predictors.append(features_sorted[ii])
            return best_predictors

class Best_features_wrap:

    def __init__(self):
        self.parameters=None


    def z_score(self,z_score_table,z_score_treshold):
        """Z_score criteria """
        "CHECKING"
        import numpy as np
        import pandas as pd

        f=lambda z: z > z_score_treshold
        z_dropped=np.mat(list(map(f,z_score_table.values)))
        Indicator=np.all(z_dropped,axis=1)
        ind_drop=np.where(Indicator==True)
        cols=z_score_table.iloc[ind_drop].index
        columns=[str(name) for name in cols]
        columns_Index=pd.Index(columns)
        columns_list=pd.Index.tolist(cols)

        if len(columns) >0:
            print("\n\n","========================================================", \
                      'There are predictors strongly significative to level  0.025 :',\
                      ','.join (str(nn) for nn in columns),
                         "========================================================", sep='\n')
        else:
             print("\n\n","========================================================", \
                      "There aren't predictors strongly significative to level  0.025 ",\
                         "========================================================", sep='\n')

        return columns_list,columns_Index
    
    def cross_validation_binary_classification(self,names,cols_data,cols_base_validation,cols_index,train_exog,\
                              x_test_exog,endog,y_test,family,\
                          method,par_reg,task,Iperfor0,K_fold,mis_K_classif):
        """Cross-validation to Binary Classification"""
        import pandas as pd
        import numpy as np
        from resources.pandas import get_pandas_from_groupby
        from resources.sets import list_minus_list
        
        x_train=pd.DataFrame(train_exog)
        y_train=pd.DataFrame(endog)
        NN=len(x_train)

        cols_base_base=cols_base_validation.copy()
        cols_index_base=cols_index.copy()

        N_features=len(cols_data) - len(cols_base_validation)

        x_train.rename(index=lambda x:x-1,inplace=True)

        index_fold=np.ones((NN,1))
        index_fold=np.cumsum(index_fold)-1
        index_fold=pd.DataFrame(index_fold)

        PP=np.cumsum(y_train)
        PP1=max(PP.iloc[:,0])
        #PP0=NN-PP1

        N_1=PP1/NN

        N_split=int(NN/K_fold)
        #N_split_last=NN-(K_fold-1)*N_split
        N_1_split=int(N_1*N_split)
        N_0_split=N_split-N_1_split

        index_0,index_1=get_pandas_from_groupby(index_fold,y_train,2)
        index_0=[int(ii) for ii in index_0]
        index_1=[int(ii) for ii in index_1]

        index_fold=index_fold.T
        i0=[]
        i1=[]
        index_test_fold=[]
        LENINDEX=0
        LENINDEX0=0
        LENINDEX1=0
        for ii in range(K_fold):
            if ii < K_fold-1:
               i0=index_0[ii*N_0_split:(ii+1)*N_0_split]
               i1=index_1[ii*N_1_split:(ii+1)*N_1_split]
               i01=sorted(np.concatenate((i0,i1)))
               LENINDEX+=len(i01)
               LENINDEX0+=len(i0)
               LENINDEX1+=len(i1)
               index_test_fold.append(i01)
            else:

               i0=index_0[(K_fold-1)*N_0_split:NN]
               i1=index_1[(K_fold-1)*N_1_split:NN]
               i01=sorted(np.concatenate((i0,i1)))
               index_test_fold.append(i01)
               LENINDEX+=len(i01)
        if (LENINDEX==index_fold.shape[1]):
            pass
        else:
            ip0=[len(index_0)-LENINDEX0 -len(i0)]
            ip1=[len(index_1)-LENINDEX1 -len(i1)]
            print("Error in computing index_test_fold, LENINDEX:",LENINDEX,ip0,ip1)
            input()
        data_fold={}
        Iperformance={}
        for ii in range(K_fold):

            index_train_fold=list_minus_list(index_fold,index_test_fold[ii])

            x_test_fold=x_train.iloc[index_test_fold[ii],:]
            y_test_fold=y_train.iloc[index_test_fold[ii],:]

            y_test_fold=np.array([ii for ii in y_test_fold.iloc[:,0]])


            x_train_fold=x_train.iloc[index_train_fold,:]
            y_train_fold=y_train.iloc[index_train_fold,:]

            y_train_fold=np.array([ii for ii in y_train_fold.iloc[:,0]])

            data_fold[ii] =[index_train_fold,x_train_fold,y_train_fold,\
                     x_test_fold,y_test_fold]

            """
            Iperfor,model_name_selected,columns_base,y_predicted_table,\
            y_estimated_table,params_table,residuals_table
            """
            Iperformance[ii]=\
            Best_features_wrap().add_feature_selection(names,cols_data,cols_base_base,\
                              cols_index_base,x_train_fold,\
                              x_test_fold,y_train_fold,y_test_fold,family,method,\
                              par_reg,task,Iperfor0,K_fold,mis_K_classif)

            #updating
            cols_base_base=cols_base_validation.copy()
            cols_index_base=cols_index.copy()

        return K_fold,N_features,data_fold,Iperformance


    def add_feature_selection(self,names,cols_data,cols_base_features,cols_index,train_exog,\
                              test_exog,endog,y_test,family,\
                          method,par_reg,task,Iperfor0,K_fold,mis_A_classif):
            """Feature and method selection based on z_score."""

            import pandas as pd
            
            from supervised.simulation_statsmodels import Statsmodels_linear_filter
            from resources.sets import list_minus_list,add_index_to_list_of_indexes
            from output.table import Table_results
            from resources.pandas import max_in_col_matrix,get_max_from_multi_index

            """
            Input:

            | names-List of models
            | Iperfor0-Percentage of performance desired
            | Iperfor -Percentage of performance obtained
            | data-data from datasets
            | cols_index-Parameter index
            | cols_base_features-cols feature to begin selection
            | U_train_exog,U_test_exog: exog from data
            | V_train,V_test :endog from data
            | Parameters to run simulation_statsmodels
            | family
            | method
            | par_reg
            | task
            |i-fold--index of fold
            |K-fold -number of folds

            Output:
            | Iperformance-Dict [Iperfor]=[model_name,cols_features]

            """

            Iperfor=0
            # Two-columns + 1
            cols_base=cols_base_features
            N=len(cols_base)
            N_base=N

            Iperformance_1={}

            while ((Iperfor <Iperfor0) or (len(cols_base) < len(cols_data)+1)):

                 columns_Index=cols_index.copy()

                 columns=cols_base.copy()


                 columns_diff=list_minus_list(cols_data,columns)


                 Iperformance_2={}

                 N_1=len(cols_data)-len(columns_diff)-2
                 #  Looking the best features on the remainder set

                 par=self.GenLogit_shape
                 
                 for nn in columns_diff:

                    columns.append(nn)
                    try:
                        columns_Index.append(nn)
                    except:
                        columns_Index=add_index_to_list_of_indexes(columns_Index,nn,True)

                    exog=train_exog[columns_Index]
                    x_test=test_exog[columns_Index]
                    

                    y_predicted_table,y_estimated_table,params_table,residuals_table,\
                    _,_,FPR,TPR,confusion_matrix,\
                    to_model_names=Statsmodels_linear_filter(names,exog,endog,x_test,y_test,par,family,method,par_reg,task,mis_A_classif).\
                    statsmodels_linear_supervised()


                    #if (Iperfor > 80):
                    # Performance by group of features and algorithms

                    Title='Confusion Matrix '+ ' with ' + str(len(columns))  + ' predictors :'+ columns[0]
                    for name in columns[1:]:
                                Title += ','+ name

                    Table_results(6,confusion_matrix,'.2f','fancy_grid',Title,60).print_table()

                    # Get a maximum performance

                    model_name_selected,model_name_list,Imax,PPV,Iperfor=\
                    max_in_col_matrix(confusion_matrix,'ACC','PPV',Iperfor)

                    # Checking len(model_name_list) > 1

                    Iperformance_2[nn]=[Imax,PPV,model_name_selected,nn,y_predicted_table,\
                                  y_estimated_table,params_table,residuals_table,columns]


                    if Iperfor > Iperfor0:
                        print("The Best performance is {0:.2f}%".format(Iperfor))

                    columns=cols_base.copy()
                    columns_Index=cols_index.copy()

                # Updating

                 N+=1

                 if (len(cols_base)==len(cols_data)):

                     break

                 Idata=pd.DataFrame(index=columns_diff,columns=['ACC','PPV'])

                 for nn in columns_diff:
                     Idata.loc[nn,'ACC']=Iperformance_2[nn][0]
                     Idata.loc[nn,'PPV']=Iperformance_2[nn][1]

                 predictor,predictor_list,Imax,PPV,Iperfor=max_in_col_matrix(Idata,'ACC','PPV',Iperfor)

                 Iperformance_1[N_1]=[Iperformance_2[predictor][0],Iperformance_2[predictor][1],Iperformance_2[predictor][8],\
                                Iperformance_2[predictor][2],\
                                Iperformance_2[predictor][4],Iperformance_2[predictor][5],\
                                Iperformance_2[predictor][6],Iperformance_2[predictor][7]]

                 # Checking len(predictor_list) > 1

                 cols_base.append(Iperformance_2[predictor][3])
                 try:
                     cols_index.append(Iperformance_2[predictor][3])
                 except:
                     cols_index=add_index_to_list_of_indexes(cols_index,Iperformance_2[predictor][3],True)

            if (K_fold==1):

                N_features=min(len(cols_data)-N_base,N_1)
                N_Idata=K_fold*N_features
                # Get the Best performance
                Idata=pd.DataFrame(index=range(N_Idata),columns=['ACC','PPV'])

                for mm in range(N_features):

                        Idata.loc[mm,'ACC']=Iperformance_1[mm][0]
                        Idata.loc[mm,'PPV']=Iperformance_1[mm][1]

                predictor,predictor_list,Imax,PPV,Iperfor=\
                max_in_col_matrix(Idata,'ACC','PPV',Iperfor0)

                if len(predictor_list)==0:
                    return print("Error in Tools.max_in_col_matrix")

                nn,mm=get_max_from_multi_index(predictor_list,N_features,1)
                Title=Iperformance_1[mm][3] + ' with ' + str(mm+3) + ' predictors :'
                Title+=','.join(str(name) for name in Iperformance_1[mm][2])

                print("The Best performance is {0:.2f}%".format(Imax) + " using", Title ,sep='\n')

                model_names=Iperformance_1[mm][3]

                columns_base=Iperformance_1[mm][2]

                y_predicted_table_selected=Iperformance_1[mm][4]

                y_estimated_table_selected=Iperformance_1[mm][5]

                params_table_selected=Iperformance_1[mm][6]

                residuals_table_selected=Iperformance_1[mm][7]


                return Iperfor,model_names,columns_base, y_predicted_table_selected,\
                            y_estimated_table_selected,params_table_selected,residuals_table_selected

            else:
                return Iperformance_1
    
    def draw_K_fold_numerical_results(self,K_fold,N_cols_base,N_features,data_fold,Iperformance):
        """...Draw K_fold performance :ACC and PPV..."""
        import pandas as pd
        from output.table import Table_results
        from output.graphics import Draw_numerical_results 

       # Get the Best performance
       # Checking the number of columns_base :2 for thr future could change
        ACC_data=pd.DataFrame(index=range(K_fold),columns=range(1+N_cols_base,1+N_cols_base+N_features))
        PPV_data=pd.DataFrame(index=range(K_fold),columns=range(1+N_cols_base,1+N_cols_base+N_features))

        for nn in range(K_fold):
            k_l=list(Iperformance[nn].keys())
            for mm in range(N_features):
                ACC_data.loc[nn,1+N_cols_base+mm]=Iperformance[nn][k_l[mm]][0]
                PPV_data.loc[nn,1+N_cols_base+mm]=Iperformance[nn][k_l[mm]][1]


        ACC_data_mean=pd.DataFrame(
                      index=['ACC Mean'],columns=range(1+N_cols_base,1+N_cols_base+N_features))
        PPV_data_mean=pd.DataFrame(
                      index=['PPV Mean'],columns=range(1+N_cols_base,1+N_cols_base+N_features))
        ACC_data_mean.loc['ACC Mean',:]=ACC_data.mean()[:]
        PPV_data_mean.loc['PPV Mean',:]=PPV_data.mean()[:]

        Table_results(10,ACC_data,'.2f','fancy_grid','ACC K_fold Cross-Validation in Binary Classification',40).print_table()

        Table_results(12,ACC_data_mean,'.2f','fancy_grid', ' ACC Data Mean vs Features', 50).print_table()

        Title="K_Fold vs Features : Cross-Validation to Binary Classification"

        # Text is not garantized inside the box draw..Why???
        Draw_numerical_results.frame_from_dict_(ACC_data,"Folds","ACC",Title,'equal',True,"ACC[ K_Fold , Number of Features]",'square')



        Table_results(11,PPV_data,'.2f','fancy_grid','PPV in K_fold Cross-Validation to Binary Classification',40).print_table()
        Table_results(13,PPV_data_mean,'.2f','fancy_grid', ' PPV Data Mean vs Features', 50).print_table()

        Title="K_Fold vs Features : Cross-Validation to Binary Classification"

        Draw_numerical_results.frame_from_dict_(PPV_data,"Folds","PPV",Title,'equal',True,"PPV [K_Fold, Number of Features]",'square')


    def K_fold_full_prediction_results(self,K_fold,N_features,data_fold,Iperformance, Iperfor0,x,y,mis_K_classif):
        """Prediction using K_fold Cros-Validation splititng Learning"""
        import pandas as pd
        from resources.pandas import max_in_col_matrix,get_max_from_multi_index
        from supervised import Statsmodels_linear_filter
        from output.table import Table_results
        from output.graphics import Draw_binary_classification_results

        N_Idata=K_fold*N_features
        # Get the Best performance
        Idata=pd.DataFrame(index=range(N_Idata),columns=['ACC','PPV'])

        for nn in range(K_fold):
            k_l=list(Iperformance[nn].keys())
            for mm in range(N_features):
                ii= N_features*nn + mm
                Idata.loc[ii,'ACC']=Iperformance[nn][k_l[mm]][0]
                Idata.loc[ii,'PPV']=Iperformance[nn][k_l[mm]][1]

        predictor,predictor_list,Imax,PPV,Iperfor=\
        max_in_col_matrix(Idata,'ACC','PPV',Iperfor0)

        for ii in predictor_list:
            nn,mm=divmod(ii,N_features)
            k_l=list(Iperformance[nn].keys())
            columns_base=Iperformance[nn][k_l[mm]][2]


            x_train_fold=data_fold[nn][1]
            y_train_fold=data_fold[nn][2]

            x_train_fold_exog=x_train_fold[columns_base]

            names=['Logit','GenLogit']
            exog=x_train_fold_exog
            endog=y_train_fold
            x_test= x[columns_base]
            #Checking
            y_test=y
            par=self.GenLogit_shape 
            family='Binomial'
            method=''
            par_reg=[]
            task="BinaryClassification"

            y_calibrated_table,y_estimated_table,params_table,residuals_table, \
            fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
            to_model_names=Statsmodels_linear_filter(names,exog,endog,x_test,y_test,par,family,method,par_reg,task,mis_K_classif).\
            statsmodels_linear_supervised()

            Table_results(0,params_table,'.3f','fancy_grid','Models Parameters',30).print_table()

            Table_results(6,confusion_matrix,'.2f','fancy_grid','Confusion Matrix ',60).print_table()

            kind="Prediction"
            Title="Prediction in Binary Classification using statsmodels"

            params=params_table.T

            Draw_binary_classification_results(FPR,TPR,names,params,\
                                x_train_fold_exog,y_train_fold,x,\
                                y,y_calibrated_table,y_estimated_table,\
                    residuals_table,columns_base,Title,kind).draw_mis_classification()

    def K_fold_numerical_results(self,K_fold,N_cols_data,N_features,data_fold,\
                                 Iperformance,Iperfor0):
        """Get numerical results from K_fold crros-validation...."""
        import pandas as pd
        from resources.pandas import max_in_col_matrix,get_max_from_multi_index
        #for ii in range(K_fold):
        #    print ('Fold=',ii)
        #    print(Iperformance[ii].keys())
        #    print("========================")

        N_Idata=K_fold*N_features
        # Get the Best performance
        Idata=pd.DataFrame(index=range(N_Idata),columns=['ACC','PPV'])

        for nn in range(K_fold):
            k_l=list(Iperformance[nn].keys())
            for mm in range(N_features):
                ii= N_features*nn + mm
                Idata.loc[ii,'ACC']=Iperformance[nn][k_l[mm]][0]
                Idata.loc[ii,'PPV']=Iperformance[nn][k_l[mm]][1]

        predictor,predictor_list,Imax,PPV,Iperfor=\
        max_in_col_matrix(Idata,'ACC','PPV',Iperfor0)

        # CHECKING THIS CRITERIA TO BEST SELECTION
        # HERE IS ADOPTED THE CRITERIA TO SELECT THE FEATURES LARGEST SUBSET

        nn,mm=get_max_from_multi_index(predictor_list,N_features,1)
        k_l=list(Iperformance[nn].keys())
        Title=Iperformance[nn][k_l[mm]][3] + ' with ' + str(mm+1+N_cols_data) + ' predictors :'
        Title+=','.join(str(name) for name in Iperformance[nn][k_l[mm]][2])
        Title+='\n '+'splitting in ' + str(K_fold) + \
        ' folds and testing the fold number: ' + str(nn)
        print("\n","The Best performance is {0:.2f}%".format(Imax) + \
              "  using", Title,"\n" ,sep='\n')


        model_name_selected=Iperformance[nn][k_l[mm]][3]

        columns_base=Iperformance[nn][k_l[mm]][2]


        x_train_fold=data_fold[nn][1]
        y_train_fold=data_fold[nn][2]

        x_test_fold=data_fold[nn][3]
        y_test_fold=data_fold[nn][4]

        x_train_fold_selected=x_train_fold[columns_base]
        x_test_fold_selected=x_test_fold[columns_base]

        y_predicted_table=Iperformance[nn][k_l[mm]][4]

        y_estimated_table=Iperformance[nn][k_l[mm]][5]

        params_table=Iperformance[nn][k_l[mm]][6]

        residuals_table=Iperformance[nn][k_l[mm]][7]

        return  Iperfor,model_name_selected,columns_base,x_train_fold_selected,y_train_fold,\
                x_test_fold_selected,y_test_fold,y_predicted_table,\
            y_estimated_table,params_table,residuals_table
