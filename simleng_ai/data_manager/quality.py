import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors
from biokit.viz import corrplot
import seaborn as sns
from scipy import stats
from ..data_manager.generation import Data_Generation
from ..data_manager.feature_eng import Data_Engineering

mpl.rcParams['figure.max_open_warning']=30

class Data_Analytics(Data_Generation):
    def __init__(self,dataset_name=None):
        self.dataset_name=dataset_name
        pass

    def data_describe(self, data):
        """Descriptive Statistics of DataFrame data."""
        from ..resources.output import table

        data_described = data.describe()
        """Grouped Data by the showing mean values"""
        data_mean = data.groupby(data.columns[-1]).mean()
        title="Descriptive Statsistics of " + self.dataset_name +" Data"
        return table(
            data_described, ".3f", "simple",title,60
        ), table(data_mean, ".3f", "simple", " Data Mean ", 60)


class Data_Quality(Data_Generation, Data_Engineering):
    def __init__(self):
        pass


class Data_Visualisation(Data_Generation):
    # colors for grahics with matplolib and plotly
    colors = [ic for ic in mcolors.BASE_COLORS.values() if ic != (1, 1, 1)]

    def __init__(self, idoc,dataset_name,title=None):
        self.idoc = idoc
        self.dataset_name = dataset_name
        self.title=str(title)
        
    def data_head_tail(self, data):
        "Head_tail of DataFrame data"
        from ..resources.output import table

        head = data.head()
        tail = data.tail()
    
        
        title_t="Data Head of " + self.dataset_name + " Dataset"
        title_h="Data Tail of " + self.dataset_name +" Dataset"
        return table(head, ".2f", "simple", title_h, 60), table(
            tail, ".2f", "simple", title_t, 60
        )

    def data_feature_show(self, data):
        from ..resources.output import image_to_report
        
        
        sns.countplot(x=data.columns[-1], data=data, palette="hls")
        plt.title("Classification: Categorical Variables")
        # return plt.show()
        if self.idoc >= 1:
            print("Fig:multiclass")
        return image_to_report(self.idoc, "multiclass", "png")

    def data_features_show(self, data):
        """check the xlabel"""
        from ..resources.output import image_to_report
        nn=1
        sns.countplot(x=data.columns[nn], hue=data.columns[-1], data=data[::5], orient="h", palette="Set1")
        plt.title("Behaviour of the " + self.dataset_name + " with the "+ data.columns[nn])
        if self.idoc >= 1:
            print("Fig:Behaviour of the "+self.dataset_name + "with the "+ data.columns[nn] )
        return image_to_report(self.idoc, "countplot", "png")
        # return plt.show()

    def data_features_draw_scatter(self, data):
        """Scatter Plot of data."""
        import seaborn as sns
        from ..resources.output import image_to_report

        sns.set(style="ticks", palette=colors[:7])
        sns.pairplot(data)
        plt.title("Scatter Plot of data")
        # return plt.show()
        if self.idoc >= 1:
            print("Fig:data_feature_scatter")
        return image_to_report(self.idoc, "scatter", "png")

    def data_features_draw_hist(self, data, n_bins):
        """Visualization of statistical distributions."""
        from ..resources.output import image_to_report

        np.random.seed(1)

        n_bins = 10
        df = data
        dfs = df.describe()
        columns = df.columns
        ncols=2
        nrows,_=np.divmod(len(columns),ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

        naxs=nrows*ncols
        
        colors = [ic for ic in mcolors.CSS4_COLORS.values() if ic != (1, 1, 1)]

        #ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axs.flatten()

        #ax = np.array([ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7])

        ax = np.array([ axi for axi in axs.flatten()])
        
        for ii in range(len(columns)):
            data = df.iloc[:, ii]
            count, mean, std, min, Q25, Q50, Q75, max = dfs.iloc[:, ii]
            bins = np.linspace(min, max, 100)
            ax[ii].hist(data, n_bins, density=True, histtype="bar", color=colors[ii])
            # ax[ii].legend(prop={'size': 10})
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            pdf = stats.norm.pdf(bin_centers, mean, std)
            ax[ii].plot(bin_centers, pdf, color=(0.3, 0.5, 0.2), lw=2)

            ax[ii].set_title(df.columns[ii])

        for ax in axs.flat[len(ax) - 1 :]:
            ax.axis("off")

        plt.suptitle(self.dataset_name + " Dataset shows multivariate distributions")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if self.idoc >= 1:
            print("Fig:data_feature_show_multiple_distributions")
        return image_to_report(self.idoc, "mult_dist", "png")

    def draw_multivariate_normal_pdf(self):
        """plot multivariate normal distribution known mean and covariance"""
        import numpy as np
        from scipy import multivariate_normal
        import matplotlib.pyplot as plt
        from ..resources.output import image_to_report

        # mean and covariance
        mu = np.array([0, 0])
        sigma = np.array([[1, -0.5], [-0.5, 1]])
        # x, y grid
        x, y = np.mgrid[-3:3:0.1, -3:3:0.1]
        X = np.stack((x.ravel(), y.ravel())).T
        norm = Tools.multivariate_normal_pdf(X, mean, sigma).reshape(x.shape)
        # Do it with scipy
        norm_scpy = multivariate_normal(mu, sigma).pdf(np.stack((x, y), axis=2))
        assert np.allclose(norm, norm_scpy)

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(
            x,
            y,
            norm,
            rstride=3,
            cstride=3,
            cmap=plt.cm.coolwarm,
            linewidth=1,
            antialiased=False,
        )
        ax.set_zlim(0, 0.2)
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.02f"))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("p(x)")
        plt.title("Bivariate Normal/Gaussian distribution")
        fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)
        # plt.show()
        if self.idoc >= 1:
            print("Fig:Multivariate Normal Distribution")
        return image_to_report(self.idoc, "multivariate", "png")


class Data_Output:
    pass


# iterative main function to evaluate the quality of data
# and generation data_output dict

# """
