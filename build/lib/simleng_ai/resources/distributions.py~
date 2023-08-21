#==["Class Multivariate","multivariate_normal_pdf","test_multivariate_normal_pdf", "get_shape_GenLogit","cdf_gensigmoid", "cdf_normalized_gensigmoid", "pdf_gensigmoid","pdf_sigmoid","cdf_sigmoid","pdf_bernoulli","cdf_bernoulli","pdf_to_cdf", "pdf_bernoulli_correction","cdf_to_pdf", "Genlogistic","lump_transf","lump_dataset","cdf_lump"]


class Multivariate_pdf:
    """
    Methods=[multivariate_normal_pdf]
    """ 
    import numpy as np
    import scipy as scp
    from scipy import stats
    import matplotlib.pyplot as plt

    def multivariate_normal_pdf(X, mean, sigma):
        """Multivariate normal probability density function over X (n_samples x n_
        ˓ → features)"""
        P = X.shape[1]
        det = np.linalg.det(sigma)
        norm_const = 1.0 / (((2*np.pi) ** (P/2)) * np.sqrt(det))
        X_mu = X - mean
        inv = np.linalg.inv(sigma)
        d2 = np.sum(np.dot(X_mu, inv) * X_mu, axis=1)
        return norm_const * np.exp(-0.5 * d2)

    def test_multivariate_normal_pdf():
        ##---TEST-----------
        # mean and covariance
        mu = np.array([0, 0])
        sigma = np.array([[1, -.5],[-.5, 1]])
        # x, y grid
        x, y = np.mgrid[-3:3:.1, -3:3:.1]
        X = np.stack((x.ravel(), y.ravel())).T
        norm = multivariate_normal_pdf(X, mu, sigma).reshape(x.shape)
        # Do it with scipy
        norm_scpy =stats.multivariate_normal(mu, sigma).pdf(np.stack((x, y), axis=2))
        assert np.allclose(norm, norm_scpy)

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(x, y, norm, rstride=3,
                           cstride=3, cmap=plt.cm.coolwarm,
                               linewidth=1, antialiased=False)
        ax.set_zlim(0, 0.2)
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('p(x)')
        plt.title('Bivariate Normal/Gaussian distribution')
        fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)
        return plt.show()

# checking for input this parameter at simglenin.txt WRONG
def get_shape_GenLogit():
        c=input("Shape parameter GenLogit: ")
        return float(c)

def cdf_gensigmoid(X,c):
        from resources.distributions import cdf_sigmoid
        f=lambda x,c :(cdf_sigmoid(x))**c
        return f(X,c).astype(float)

def cdf_normalized_gensigmoid(X,c):
        import numpy as np
        z=np.exp(X)**c
        f=lambda x,c:z/(1 + sum(z))
        return f(X,c).astype(float)

def pdf_gensigmoid(X,c):
        import numpy as np
        f=lambda x,c: c*np.exp(-x)/(1+np.exp(-x))**(c+1)
        return f(X,c).astype(float)
    # Mapping training and testing data to [0,1] X->U
def cdf_sigmoid(X):
        import numpy as np
        f=lambda x:1/(1+ np.exp(-x))
        return f(X).astype(float)

def pdf_sigmoid(X):
        import numpy as np
        f=lambda x:np.exp(-x)/(1+ np.exp(-x))**2
        return f(X).astype(float)

def pdf_bernoulli(X,y,c):
        from resources.distributions import cdf_sigmoid,pdf_sigmoid
        import numpy as np
        z_cdf=cdf_gensigmoid(X-0.5,c) #must be taken an original cdf
        z_pdf=pdf_gensigmoid(X,c) #must be taken an original pdf
        f=lambda i:np.power(z_cdf[i],2*y[i]-1)*np.power(z_pdf[i],(1-y[i]))
        #f=lambda i:np.power(z_cdf[i],y[i])*np.power(1-z_cdf[i],1-y[i])
        aa=list(map(f,range(len(y))))
        ll_h=np.asarray(aa).astype(float)
        return z_cdf,z_pdf,ll_h

def cdf_bernoulli(X,y,c):
        from resources.distributions import pdf_to_cdf,pdf_bernoulli_correction 
        n=X.shape[0]
        X_sorted,y_sorted=array_sort_y_on_x(X,y,0)
        _,_,pdf_sorted=pdf_bernoulli(X_sorted,y_sorted,c)
        pdf=pdf_bernoulli_correction(pdf_sorted,y_sorted)
        cdf=pdf_to_cdf(pdf,len(pdf))
        X_sorted.reshape(n,1)
        y_sorted.reshape(n,1)
        pdf.reshape(n,1)
        cdf.reshape(n,1)
        return X_sorted,y_sorted,pdf,cdf
    
def pdf_to_cdf(X,dim):
        """ Discrete random variable"""
        import numpy as np
        sumx=sum(X)
        cdf=[ ]
        for ii in range(1,dim+1):
            cdf.append(sum(X[:ii])/sumx)
        cdf=np.asarray(cdf).astype(float)
        cdf.reshape(dim,1)
        #pdf=Tools.cdf_to_pdf(cdf,sumx,dim)
        #for ii in range(dim):
         #   print(X[ii],cdf[ii],pdf[ii])
        return cdf

def pdf_bernoulli_correction(X,y):
        import numpy as np
        n=X.shape[0]
        pdf=[]
        eps=1e-010
        for i in range(n):
            if (y[i]==0 and X[i]>=0.5 + eps):
                     pdf.append(X[i]/2)
            elif (y[i]==1 and X[i]<=0.5 - eps):
                     pdf.append(2*X[i])
            else:
                pdf.append(X[i])
        pdf=np.asarray(pdf).astype(float)
        #pdf.reshape(n,1)
        return pdf


def cdf_to_pdf(X,scale,dim):
        """ Discrete random variable"""
        import numpy as np
        pdf=[X[0]*scale]
        for ii in range(dim-1):
            pdf.append((X[ii+1]-X[ii])*scale)
        pdf=np.asarray(pdf).astype(float)
        return pdf

    
def Genlogistic(Nx,Nc):
        import pandas as pd
        import numpy as np
        from resources.distributions import cdf_gensigmoid,pdf_gensigmoid
        from output.graphics import Draw_numerical_results
        x=np.linspace(-20,20,Nx)
        C_shape=np.linspace(0.1,1.0,Nc)

        cdf=pd.DataFrame(np.empty((Nx,Nc)),index=x,columns=C_shape)
        pdf=pd.DataFrame(np.empty((Nx,Nc)),index=x,columns=C_shape)
        for c in C_shape:
            yy=cdf_gensigmoid(x,c)
            cdf.loc[:,c]=yy
            zz=pdf_gensigmoid(x,c)
            pdf.loc[:,c]=zz
            pdf_max=pdf.max()
            pdf=pdf/pdf_max
        Draw_numerical_results.frame_from_dict(x,pdf,'x-values','pdf','Genlogistic pdf [c-shape values]',\
                'equal',False,'','square',0)
        Draw_numerical_results.frame_from_dict(x,cdf,'x-values','cdf','Genlogistic cdf [c-shape values]',
                'equal',False,'','square',0)
        return pdf,cdf
    #New Method

def lump_transf(i,X,par):
        from resources.algebra import add_data,prod_data,mean_rows_data,dot_vector_rows_data
        if i==0:
            X=add_data(X)
        elif i==1:
            X=prod_data(X)
        elif i==2:
            X=mean_rows_data(X)
        else:
            X=dot_vector_rows_data(par,X)
        return X.T

def lump_dataset(X,n,p,transf,par):
        from resources.distributions import lump_transf
        """Lumping method of datasets
        x-Dataset
        n-sample size
        p-Features subset size
        ind-index of transformation
        0-sum
        1-mulitplicative
        2-average
        3-optim-linear combinations(optimization parameters)
        PAR-optimized vector of parameters
        """
        import numpy as np
        X=np.array(X)
        n,m=X.shape
        if transf=='sum':ind=0
        elif transf=='prod':ind=1
        elif transf=='average':ind=2
        else:
            ind=3
        return lump_transf(ind,X,par)

def cdf_lump(X,n,y,family,cdf_par):
        from resources.distributions import cdf_bernoulli
        """Get values of probability of X based on cdf of a family
        |X-(1,n) vector
        |n-vector dimension
        |y-labels of training data sets
        |family-pdf to selected
        |0-Logit
        |1-Genlogit(c-shape-values)
        |cdf-par : c-shape parameter of GenLogit
        Output:
           Xnew=cdf(X)
           Bi(X)=cdf(x)**y*(1-cdf(x))**(1-y) Bernoulli event
        """
        return cdf_bernoulli(X,y,cdf_par)

  
