import os
import math
import random
import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt


class Knn():
    def __init__(self, data, groups=2):
        self.clusters = groups
        self.feature = data
        self.rows = data.shape[0]
        self.columns = data.shape[1]
        self.centroids = random.sample(list(self.feature), groups)
        self.previous_centroids = np.empty((groups, data.shape[1]))
        self.centroid_idx = np.zeros((self.rows, 1), dtype=int)

    def fit(self):
        norms = self.find_distance()
        self.centroid_idx = np.argmin(norms, axis=1)

        if self.feature.shape[1] == 2:
            self.plot2D_data()

        centroid_means = self.find_mean()

    def find_distance(self):
        temp = np.zeros((self.rows, self.clusters))
        for i in range(self.clusters):
            temp[:, i] = np.sqrt(np.sum(
                                 np.square(self.feature - self.centroids[i]),
                                 axis=1))
        return temp

    def find_mean(self):
        self.previous_centroids = self.centroids[:]
        for i in range(self.clusters):
            self.centroids[i] = np.mean(self.feature[self.centroid_idx == i],
                                        axis=1)
            print(self.centroids[i])

    def plot2D_data(self):
        plt.figure(1)
        plt.scatter(self.feature[:, 0], self.feature[:, 1],
                    c="red", s=15)
        for i in range(self.clusters):
            plt.scatter(self.centroids[i][0], self.centroids[i][1],
                        c="blue", alpha=0.8, marker="x", s=40)
        plt.figure(2)
        plt.hist(self.centroid_idx)
        plt.show()


X = scipy.io.loadmat(os.path.join(os.getcwd(), "src", "ex7data2.mat"))
X = X['X']
obj = Knn(X, 3)
obj.fit()
