from typing import NoReturn

import numpy as np
from sklearn.neighbors import KDTree
from utils import init_k_means_pp, init_random, init_sample


class KMeans:
    def __init__(self, n_clusters: int, init: str = "k-means++", max_iter: int = 300):
        self.clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

    def fit(self, X: np.array, y=None) -> NoReturn:
        if self.init == "random":
            print(X[0].shape)
            self.centroids = init_random(self.clusters, X[0].shape[0])
        if self.init == "sample":
            self.centroids = init_sample(X, self.clusters)
        if self.init == "k-means++":
            self.centroids = init_k_means_pp(X, self.clusters)

    def predict(self, X: np.array) -> np.array:
        clusters_ind = []
        for i in range(self.max_iter):
            tree = KDTree(self.centroids, leaf_size=1)
            clusters_ind = np.squeeze(tree.query(X, k=1, return_distance=False).T)
            new_centroids = np.array(
                [X[clusters_ind == [i]].mean(axis=1) for i in range(self.clusters)],
                dtype=object,
            )

            not_empty_clusters = np.unique(clusters_ind)
            if not_empty_clusters.shape[0] != self.clusters:
                for cluster_index in range(0, self.clusters):
                    if not (cluster_index in not_empty_clusters):
                        self.centroids[cluster_index] = np.rand(X[0].shape[0])
                        i = i - 1
                        continue

            if np.array_equal(new_centroids, self.centroids):
                return clusters_ind
        return clusters_ind
