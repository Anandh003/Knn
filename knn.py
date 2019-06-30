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
        np.random.shuffle(data)
        self.centroids = data[0:groups,:]
        self.previous_centroids = self.centroids[:]
        self.centroid_idx = np.zeros((self.rows, 1), dtype=int)

    def fit(self):
        limit = 0
        plot_data = False
        max_limit = 15
        if self.feature.shape[1] == 2:
            plot_data = True
            self.plot2D_data()
        i = 0
        while(i <= max_limit):
            i += 1     
            norms = self.find_distance()
            self.centroid_idx = np.argmin(norms, axis=1)
            self.find_mean()
            if plot_data:
                self.plot2D_data()
            if np.array_equal(self.previous_centroids,
                              self.centroids) and limit >= 4:
                limit += 1
                break
        print("Previous Centroids : ", self.previous_centroids)

    def find_distance(self):
        temp = np.zeros((self.rows, self.clusters))
        for i in range(self.clusters):
            temp[:, i] = np.sqrt(np.sum(
                                 np.square(self.feature - self.centroids[i]),
                                 axis=1))
        return temp

    def find_mean(self):
        for i in range(self.clusters):
            self.centroids[i] = np.mean(self.feature[self.centroid_idx == i],
                                        axis=0)
        print(self.centroids)
        self.previous_centroids = np.concatenate(
                                    (self.previous_centroids, self.centroids)
                                    )
        return self.centroids

    def plot2D_data(self):
        plt.figure(1)
        plt.scatter(self.feature[:, 0], self.feature[:, 1],
                    c="red", s=15)
        plt.scatter(self.centroids[:,0], self.centroids[:,1],
                        c="blue", alpha=0.8, marker="x", s=40)
        plt.draw()
        plt.pause(1)
        plt.clf()


X = scipy.io.loadmat(os.path.join(os.getcwd(), "src", "ex7data2.mat"))
X = X['X']
obj = Knn(X, 3)
obj.fit()
