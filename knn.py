import os
import math
import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt


# def plot_data(x, y, kind="plot",
#               align="center",
#               x_axis="X_axis",
#               y_axis="Y_axis",
#               title="Graph",
#               color="r",
#               marker="x",
#               size=20,
#               label="y"):
#     plt.ion()
#     if kind == "scatter":
#         plt.scatter(x, y,
#                     color=color,
#                     label=label,
#                     s=size,
#                     marker=marker)
#     elif kind == "plot":
#         plt.plot(x, y,
#                  color=color,
#                  marker=marker,
#                  label=label,
#                  markersize=size)
#     else:
#         raise ValueError("Kind: {} is not valid".format(kind))
#     plt.xlabel(x_axis)
#     plt.ylabel(y_axis)
#     plt.title(title)
    # plt.show()
class Knn():
    def __init__(self,data,groups=2):
        self.clusters = groups
        self.feature = data
        self.rows = data.shape[0]
        self.columns = data.shape[1]
        self.centroids = data.sample(n=groups)
        self.centroid_idx = pd.DataFrame(np.zeros((self.rows,1), dtype=int))

    def fit(self):
        distances = np.zeros(self.rows, self.groups)
        for i in range(self.clusters):
            for j in range(self.rows):
                distances[j,i] = math.sqrt(centroids[j:]) 


# def create_centroids():


X = scipy.io.loadmat(os.path.join(os.getcwd(), "src", "ex7data2.mat"))
X = pd.DataFrame(X['X'])
obj = Knn(X, 3)
