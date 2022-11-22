import sys

import numpy as np


class AgglomertiveClustering:
    def distance(self, p1, p2):
        return np.sum((p1 - p2) ** 2)

    def check_dist_between_clusters_precalc(self, X: np.array, a: int, b: int):
        points_a = X[self.clusters == a]
        points_b = X[self.clusters == b]
        if points_a.size == 0 or points_b.size == 0:
            return None
        dist = np.array([((points_b - a) ** 2).sum(axis=1) for a in points_a])
        if self.linkage == 'average':
            return dist.mean()
        if self.linkage == 'single':
            return dist.min()
        return dist.max()

    def check_dist_between_clusters(self, X: np.array, a: int, b: int):
        points_a_ind = np.where(self.clusters == a)[0]
        points_b_ind = np.where(self.clusters == b)[0]
        if points_a_ind.size == 0 or points_b_ind.size == 0:
            return None
        dist = self.dist[np.ix_(points_a_ind, points_b_ind)]
        if self.linkage == 'average':
            return dist.mean()
        if self.linkage == 'single':
            return dist.min()
        return dist.max()

    def __init__(self, n_clusters: int = 16, linkage: str = "single"):
        self.clusters_count = n_clusters
        self.linkage = linkage
        self.dist = np.zeros((1, 1))

    def fit_predict(self, X: np.array, y=None) -> np.array:
        self.clusters = np.arange(X.shape[0])
        self.clusters_map = set(range(X.shape[0]))
        self.dist = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i):
                self.dist[i][j] = self.dist[j][i] = self.distance(X[i], X[j])
        amount_clusters = X.shape[0]
        while amount_clusters > self.clusters_count:
            min_dist = sys.maxsize
            for i in range(X.shape[0]):
                for j in range(i):
                    if not i in self.clusters_map:
                        continue
                    if not j in self.clusters_map:
                        continue
                    if self.dist[i, j] is None:
                        continue
                    dist_cluster = self.dist[i, j]
                    if dist_cluster < min_dist:
                        min_dist = dist_cluster
                        a = i
                        b = j
            self.clusters[self.clusters == a] = b
            self.clusters_map.remove(a)
            for i in range(X.shape[0]):
                if i == b:
                    continue
                if not i in self.clusters:
                    continue
                self.dist[b][i] = self.check_dist_between_clusters(X, b, i)
                self.dist[i][b] = self.dist[b][i]
            amount_clusters -= 1
        unique_clusters = np.unique(self.clusters)
        for i in range(self.clusters_count):
            self.clusters[self.clusters == unique_clusters[i]] = i
        return self.clusters
