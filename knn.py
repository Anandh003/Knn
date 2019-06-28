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
        self.centroid_idx = np.zeros((self.rows, 1), dtype=int)

    def fit(self):
        norms = self.find_distance()
        self.centroid_idx = np.argmin(norms, axis=1)

    def find_distance(self):
        temp = np.zeros((self.rows, self.clusters))
        for i in range(self.clusters):
            temp[:, i] = np.sqrt(np.sum(np.square(self.feature - i), axis=1))
        return temp

X = scipy.io.loadmat(os.path.join(os.getcwd(), "src", "ex7data2.mat"))
X = X['X']
obj = Knn(X, 3)
obj.fit()
