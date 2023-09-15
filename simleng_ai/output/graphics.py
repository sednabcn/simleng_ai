# colors for grahics with matplolib and plotly
from matplotlib import colors as mcolors
from ..resources.output import image_to_report

colors = [ic for ic in mcolors.BASE_COLORS.values() if ic != (1, 1, 1)]
from collections import OrderedDict


class Draw_numerical_results:
    """Draw numerical properties."""

    @staticmethod
    def frame_from_dict(
        x, y, xlabel, ylabel, Title, mapping, grid, text, boxstyle, mode
    ):
        import numpy as np
        import pandas as pd

        mode = mode
        boxstyle = boxstyle
        """From dict with orient='index'"""
        m, n = y.shape
        X = np.array(np.ones((m, n)))

        if X.all() == None:
            X = np.array(X)
            X = pd.DataFrame(X)
            X.loc[0, :] = 0
            X = np.cumsum(X)
        else:
            for ii in range(n):
                X[:, ii] = np.copy(y.index)
            X = pd.DataFrame(X)

        Labels = y.columns.values

        if mapping == "Log":
            Title = "Log20 " + Title
            Y = np.log(np.abs(y)) / np.log(20)
        else:
            Y = y
        Title = Title
        xlabel = xlabel
        ylabel = ylabel
        linestyles = OrderedDict(
            [
                ("solid", (0, ())),
                ("mmmmm", (0, (5, 1, 1, 5, 5, 1))),
                ("nnnnn", (0, (5, 1, 5, 1, 5, 1))),
                ("densely dotted", (0, (1, 1))),
                ("loosely dashed", (0, (5, 10))),
                ("dashed", (0, (5, 5))),
                ("densely dashed", (0, (5, 1))),
                ("loosely dashdotted", (0, (3, 10, 1, 10))),
                ("dashdotted", (0, (3, 5, 1, 5))),
                ("densely dashdotted", (0, (3, 1, 1, 1))),
                ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
                ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
                ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
                ("xxxx", (0, (3, 1, 5, 1, 1, 5))),
                ("yyyy", (0, (5, 1, 1, 1, 5, 5))),
                ("zzzzzzzzzzzzzzzzz", (0, (3, 5, 1, 5, 1, 5))),
                ("loosely dotted", (0, (1, 10))),
                ("dotted", (0, (1, 5))),
            ]
        )

        Linestyle = linestyles.values()

        return Drawing2d.plot_matrix_matrix(
            X,
            Y,
            Title,
            xlabel,
            ylabel,
            Labels,
            Linestyle,
            0,
            mapping,
            grid,
            text,
            boxstyle,
            mode,
        )

    def frame_from_dict_(y, xlabel, ylabel, Title, mapping, grid, text, boxstyle, mode):
        """From dict with orient='index'"""
        import numpy as np
        import pandas as pd

        m, n = y.shape
        X = np.array(np.ones((m, n)))
        X = np.array(X)
        X = pd.DataFrame(X)

        X.loc[0, :] = 0
        X = np.cumsum(X)
        Labels = y.columns.values
        if mapping == "Log":
            Title = "Log20 " + Title
            Y = np.log(np.abs(y)) / np.log(20)
        else:
            Y = y
        Title = Title
        xlabel = xlabel
        ylabel = ylabel
        Linestyle = ["-", "-.", "--", ":", "--"]

        boxstyle = boxstyle

        return Drawing2d.plot_matrix_matrix(
            X,
            Y,
            Title,
            xlabel,
            ylabel,
            Labels,
            Linestyle,
            0,
            mapping,
            grid,
            text,
            boxstyle,
            mode,
        )


class Drawing2d:
    """Draw 2d figures"""

    def __init__(self):
        self.parameters = None

    def plot_matrix_matrix(
        X,
        Y,
        Title,
        xlabel,
        ylabel,
        Labels,
        Linestyle,
        kind,
        scale,
        grid,
        text,
        boxstyle,
        mode=None,
    ):
        """Draw matrix vs matrix."""
        """
        kind of graphic
        0-plot
        1-scatter
        2-stem
        scale of plot
        'equal'
        'Log'
        'semilogy'
        'semilogx'
        'Loglog'
        """
        import random
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        from ..resources.output import image_to_report

        text = text
        # Check the parameter grid
        grid = grid

        if len(Y.columns) > 7:
            colors = [ic for ic in mcolors.CSS4_COLORS.values() if ic != (1, 1, 1)]
            keys = [
                kc
                for kc in mcolors.CSS4_COLORS.keys()
                if mcolors.CSS4_COLORS[kc] != (1, 1, 1)
            ]
            linewidth = 2
        else:
            colors = [ic for ic in mcolors.BASE_COLORS.values() if ic != (1, 1, 1)]
            keys = [
                kc
                for kc in mcolors.BASE_COLORS.keys()
                if mcolors.BASE_COLORS[kc] != (1, 1, 1)
            ]
            linewidth = 2

        # colors for grahics with matplolib and plotly
        # colors = [ic for ic in mcolors.BASE_COLORS.values() if ic !=(1,1,1)]
        # keys   = [kc  for kc in mcolors.BASE_COLORS.keys() if mcolors.BASE_COLORS[kc]!=(1,1,1)]

        """
        Checking orders of matrices
        """
        if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
            print("These matrices are not equal order")

        idx = [X.iloc[:, i].argsort() for i in range(X.shape[1])]
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        for i, linestyle in enumerate(Linestyle):
            if i in range(X.shape[1]):
                label = Labels[i]

                if kind == 0:
                    if scale == "equal" or "Log":
                        ax.plot(
                            X.iloc[idx[i], i],
                            Y.iloc[idx[i], i],
                            linestyle=linestyle,
                            linewidth=1.5,
                            label=label,
                            color=colors[i],
                        )
                    elif scale == "semilogy":
                        ax.semilogy(
                            X.iloc[idx[i], i],
                            Y.iloc[idx[i], i],
                            linestyle=linestyle,
                            linewidth=1.5,
                            label=label,
                            color=colors[i],
                        )
                    elif scale == "semilogx":
                        ax.semilogx(
                            X.iloc[idx[i], i],
                            Y.iloc[idx[i], i],
                            linestyle=linestyle,
                            linewidth=1.5,
                            label=label,
                            color=colors[i],
                        )
                    elif scale == "Loglog":
                        ax.loglog(
                            X.iloc[idx[i], i],
                            Y.iloc[idx[i], i],
                            linestyle=linestyle,
                            linewidth=1.5,
                            label=label,
                            color=colors[i],
                        )
                    else:
                        pass
                elif kind == 1:
                    ax.scatter(
                        X.iloc[idx[i], i],
                        Y.iloc[idx[i], i],
                        marker=linestyle,
                        label=label,
                        color=colors[i],
                    )
                elif kind == 2:
                    markerline, stemlines, baseline = ax.stem(
                        X.iloc[idx[i], i],
                        Y.iloc[idx[i], i],
                        "-.",
                        markerfmt=keys[i] + "o",
                        label=label,
                    )
                    # ax.setp(baseline, color='r', linewidth=1.5)
                else:
                    pass
            else:
                pass

        ax.legend(prop={"size": 10}, loc=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(Title)
        plt.tight_layout(rect=[0, 0, 1, 1])

        if len(Y.columns) > 7:
            ax.legend(
                prop={"size": 10},
                loc=1,
                mode="expand",
                numpoints=1,
                ncol=10,
                fancybox=True,
                fontsize="small",
            )

        if grid == False:
            plt.grid(grid)
        else:
            plt.grid(grid, color=(1, 0, 0), linestyle="", linewidth="1")

        if len(text) > 0:
            # Check the location of text
            plt.text(
                0.3,
                75.0,
                text,
                {
                    "color": "k",
                    "fontsize": 10,
                    "ha": "left",
                    "va": "center",
                    "bbox": dict(boxstyle=str(boxstyle), fc="w", ec="k", pad=0.3),
                },
            )
        # plt.grid(grid)
        # plt.grid(color=colors[i+1],linestyle='',linewidth='1')
        # if len(text)>0:
        #      plt.text(0.1, 85.0,text,\
        #             {'color': 'k', 'fontsize':10, 'ha': 'left', 'va': 'center',\
        #              'bbox': dict(boxstyle=str(boxstyle), fc="w", ec="k", pad=0.3)})

        # return plt.show()
        return image_to_report(mode, Title[:3] + "_" + str(random.randint(0, 30)), "png")

    def draw_vector(v0, v1, x0, y0, ax=None):
        """Modification to Python Data Science Handbook
        origin by Jake VanderPlas; Jupyter notebooks"""
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        arrowprops = dict(
            facecolor="black", arrowstyle="->", linewidth=2, shrinkA=0, shrinkB=0
        )
        ax.annotate("", xy=(v0, v1), xytext=(x0, y0), arrowprops=arrowprops)


class Draw_binary_classification_results:
    """Draw the results."""

    """
     0. Roc_curve

    """

    def __init__(
        self,
        FPR,
        TPR,
        model_names,
        params,
        exog,
        endog,
        x,
        y,
        y_predict,
        y_estimated,
        residuals,
        columns,
        Title,
        kind,
        idoc,
        shape=None,    
            
    ):
        self.fpr = FPR
        self.tpr = TPR
        self.model_names_selected = model_names
        self.params = params
        self.x_train = exog
        self.y_train = endog
        self.x_test = x
        self.y_test = y
        self.y_predict = y_predict
        self.y_estimated = y_estimated
        self.residuals = residuals
        self.columns = columns
        self.Title = Title
        self.kind = kind
        self.idoc = idoc
        self.GenLogit_shape=shape
        
    def roc_curve(self):
        X = self.fpr.T
        Y = self.tpr.T
        Title = "Roc Curve "
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        Labels = self.model_names_selected
        Linestyle = ["-", "-.", "--", ":", "--"]
        return Drawing2d.plot_matrix_matrix(
            X,
            Y,
            Title,
            xlabel,
            ylabel,
            Labels,
            Linestyle,
            0,
            "equal",
            "",
            "",
            "",
            self.idoc,
        )

    def fpn_estimated(self):
        """Draw prediction results from binary classification..."""
        import numpy as np
        import pandas as pd

        m, n = self.y_estimated.shape
        X = np.array(np.ones((m, n)))
        X = np.array(X)
        X = pd.DataFrame(X)
        X.loc[:, 0] = 0
        X = np.cumsum(X)
        y_test = np.array([self.y_test]).T
        Labels = self.y_estimated.columns.values
        # y=np.concatenate([y_truth,y_predict],axis=1)
        f = lambda x: np.where(x < 0.5, 0, 1)
        y_estimated = [list(map(f, self.y_estimated.iloc[:, i])) for i in range(n)]
        y_estimated = pd.DataFrame(y_estimated).T
        y = y_test - y_estimated
        y = pd.DataFrame(y)
        Y = y
        # Title='Binary Classification using Statsmodels'
        Title = self.Title
        xlabel = "samples"
        ylabel = "y_truth - y_estimated"
        Linestyle = ["-", "-.", "--", ":", "--"]
        return Drawing2d.plot_matrix_matrix(
            X,
            Y,
            Title,
            xlabel,
            ylabel,
            Labels,
            Linestyle,
            2,
            "equal",
            "",
            "",
            "",
            self.idoc,
        )

    def draw_regions_binary_classification(self):
        """Draw regions of binary classification."""
        """
        | x--x_test
        | y--y_test
        | y_estimated =model.predict(x_test)
        | columns --all columns including x and y
        | treshold--vector contains treshold to each models
        | c -shape parameter GenLogit
        """
        from ..resources.distributions import get_shape_GenLogit
        from ..resources.output import image_to_report
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        #print(self.idoc)
        #print(input("STOP"))

        model_names = self.model_names_selected

        self.x_test = pd.DataFrame(self.x_test)
        self.y_test = pd.DataFrame(self.y_test)
        self.y_estimated = pd.DataFrame(self.y_estimated)

        MM = len(self.model_names_selected)

        z_list = []
        zz = pd.DataFrame(
            np.concatenate([self.x_test, self.y_test], axis=1), columns=self.columns
        )
        z_list.append(zz)

        for ii in range(MM):
            y_estimated_col = self.y_estimated.iloc[:, ii]
            y_estimated_col = np.where(y_estimated_col < 0.5, 0, 1)
            y_estimated_col = np.array(y_estimated_col).reshape(self.x_test.shape[0], 1)
            zz = pd.DataFrame(
                np.concatenate([self.x_test, y_estimated_col], axis=1),
                columns=self.columns,
            )
            z_list.append(zz)

        # Compute the coeficients of straight line as boundary_decision
        treshold = []
        for name in model_names:
            if name == "smd.Logit" or "sm.GLM" or "MNLogit":
                treshold.append(0.0)
            elif name == "GenLogit":
                # checking for input this parameter at simglenin.txt
                #c = get_shape_GenLogit()
                c = self.GenLogit_shape
                treshold.append(0.5**float(c))

            else:
                treshold.append(0.5)

        model_names.reverse()
        model_names.append("Test")
        model_names.reverse()

        """N-Number of subplots..."""
        N = len(model_names)
        if N % 2 == 0:
            nrows = int(N / 2)
            ncols = 2
            figsize = (15, 10)
        elif N % 3 == 0:
            nrows = int(N / 3)
            ncols = 3
            figsize = (16, 4)
        elif N % 4 == 0:
            nrows = int(N / 4)
            ncols = 4
            figsize = (8, 4)
        else:
            nrows = int(N)
            ncols = 1
            figsize = (10, 15)
            # Don't working plt.sub_plots_adjust
            #   wspace=0.5
            #  hspace=0.5

        # Draw scatter plots of "glu" vs "ped"

        # Title="Binary Classification Regions (Yes/No) using statsmodels"
        Title = self.Title
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for ax, df, name in zip(axes.flatten(), z_list, model_names):
            binary=df.groupby(df.columns[-1])
            b_=list(binary.apply(list).keys())

            for color, dfe, nlabel in zip(
                ["Darkblue", "red"],binary , [str(b_[0]), str(b_[1])]
            ):
                # x=df.loc[Index,df.columns[0]]
                x = dfe[1][df.columns[0]]
                y = dfe[1][df.columns[1]]

                ax.scatter(x=x, y=y, marker="o", c=color, s=70, label=str(nlabel))
                ax.set_title(name)
                ax.set_xlabel(df.columns[0])
                ax.set_ylabel(df.columns[1])
                ax.legend()

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(Title)
        # return plt.show()
        return image_to_report(self.idoc, "binaryclass", "png")

    def draw_mis_classification(self):
        """Draw regions of binary classification."""
        """
        YOU HAVE TO REWRITE THIS ONE
        | x--x_test
        | y--y_test
        | y_estimated =model.predict(x_test)
        | columns --all columns including x and y
        | treshold--vector contains treshold to each models
        | c -shape parameter GenLogit
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from ..resources.output import image_to_report
        from ..resources.distributions import get_shape_GenLogit
        
        
        model_names = self.model_names_selected

        if self.kind == "Validation":
            # y_train=pd.DataFrame(self.y_train)
            # y_predict_val=pd.DataFrame(self.y_predict)
            # residuals_table=self.residuals
            NN = len(self.y_train)
            x_graph_val = np.linspace(0, NN - 1, NN).astype(float)

        elif self.kind == "Prediction":
            # y_test=pd.DataFrame(self.y_test)
            # y_predict_est=pd.DataFrame(self.y_estimated)
            NN = len(self.y_test)
            x_graph_est = np.linspace(0, NN - 1, NN).astype(float)
        else:
            print("Error on kind selection")
        treshold=[]
        z_list = []
        for name in model_names:
            if name == "Logit":
                sname= "smd.Logit"

            elif name=="Probit":
                sname="smd.Probit"
                treshold.append(0.0)
            elif name =="GenLogit":
                 sname="GenLogit"
                 #c = get_shape_GenLogit()
                 c=self.GenLogit_shape
                 treshold.append(0.5**float(c))
            else:
                treshold.append(0.5)

            if self.kind == "Validation":
                
                z_ordinates_ii = np.abs(self.residuals[sname])

                # z_predict_ii=y_predict_val[sname]

                z_predict_ii = self.y_predict[sname]

                z_predict_val = np.where(z_predict_ii < 0.5, 0, 1)

                # z_predict_val=pd.DataFrame(z_predict_val)

                z_res_val = np.vstack([x_graph_val, z_ordinates_ii]).T

                z_ind_val = self.y_train - z_predict_val

                zz = pd.DataFrame(z_res_val, columns=["obs", "residue_pearson"])
                zz["type"] = z_ind_val

            elif self.kind == "Prediction":
                z_ordinates_ii = np.abs(self.y_estimated[sname])

                # z_predict_ii=y_predict[sname]

                # z_predict=np.where(z_predict_ii<0.5,0,1)

                # z_predict=pd.DataFrame(z_predict)

                z_est = np.vstack([x_graph_est, z_ordinates_ii]).T

                # z_ind_test=y_test-z_predict

                zz = pd.DataFrame(z_est, columns=["obs", "prediction_values"])
                zz["type"] = self.y_test
            else:
                print("Error on kind selection")
            z_list.append(zz)

        N = len(model_names)
        nrows = int(N)
        ncols = 1
        figsize = (10, 5)
        # wspace=0.5
        # hspace=0.5

        Title = self.Title
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for ax, df, name in zip(axes.flatten(), z_list, model_names):
            ind = sorted(df[df.columns[-1]])[0]
            if ind == 0:
                Color = ["red", "green"]
                LABEL = ["TP+TN", "FN"]
            elif ind == 1:
                Color = ["green"]
                LABEL = ["FN"]
            else:
                Color = ["Darkblue", "red", "green"]
                LABEL = ["FP", "TP+TN", "FN"]

            for color, dfe, nlabel in zip(Color, df.groupby(df.columns[-1]), LABEL):
                # x=df.loc[Index,df.columns[0]]
                x = dfe[1][df.columns[0]]
                y = dfe[1][df.columns[1]]

                ax.scatter(x=x, y=y, marker="o", c=color, s=40, label=str(nlabel))

                ax.set_xlim([0, NN])
                ax.set_ylim([0, 3])
                ax.set_title(name)
                ax.set_xlabel(df.columns[0])
                ax.set_ylabel(df.columns[1])

                ax.legend(loc="best")
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.suptitle(Title)
        # return plt.show()
        return image_to_report(self.idoc, "missclass", "png")
