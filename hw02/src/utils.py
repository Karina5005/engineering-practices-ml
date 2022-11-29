from typing import NoReturn

import cv2
import numpy as np
from matplotlib import pyplot as plt

from agglomertive import AgglomertiveClustering


def visualize_clasters(X, labels, plt_name):
    unique_labels = np.unique(labels)
    unique_colors = np.random.random((len(unique_labels), 3))
    colors = [unique_colors[lab] for lab in labels]
    plt.figure(figsize=(9, 9))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.savefig(f"results/{plt_name}")


def clusters_statistics(flatten_image, cluster_colors, cluster_labels, plt_name):
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    for remove_color in range(3):
        axes_pair = axes[remove_color]
        first_color = 0 if remove_color != 0 else 2
        second_color = 1 if remove_color != 1 else 2
        axes_pair[0].scatter(
            [p[first_color] for p in flatten_image],
            [p[second_color] for p in flatten_image],
            c=flatten_image,
            marker=".",
        )
        axes_pair[1].scatter(
            [p[first_color] for p in flatten_image],
            [p[second_color] for p in flatten_image],
            c=[cluster_colors[c] for c in cluster_labels],
            marker=".",
        )
        for a in axes_pair:
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
    plt.savefig(f"results/{plt_name}")


def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


def init_k_means_pp(X: np.array, k: int):
    centroids = list(init_sample(X, 1))
    for c_id in range(k - 1):
        dist = []
        for i in range(X.shape[0]):
            point = X[i, :]
            d = distance(point, centroids[0])
            for j in range(len(centroids)):
                d = min(d, distance(point, centroids[j]))
            dist.append(d)
        dist = np.array(dist)
        next_centroid = X[np.argmax(dist), :]
        centroids.append(next_centroid)
    return np.array(centroids)


def init_random(k: int, d: int):
    return np.random.rand(k, d)


def init_sample(X: np.array, k: int):
    a = np.zeros((X.shape[0],))
    a[0:k] = 1
    np.random.shuffle(a)
    return X[a == 1]


def read_image(path: str) -> np.array:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image / 255


def save_image(image: np.array, path: str) -> NoReturn:
    cv2.imwrite(f"results/{path}", image)


def clusterize_image(image, **kwargs):
    image_compr = np.unique(
        np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2])), axis=0
    )
    cluster_colors = np.random.random((50, 3))  # color of each cluster
    agg_clustering = AgglomertiveClustering(n_clusters=50, linkage="complete")
    clusters = agg_clustering.fit_predict(
        image_compr
    )  # Cluster labels for each pixel in flattened image
    recolored = image.copy()
    for i in np.unique(clusters):
        recolored[
            np.isin(recolored, image_compr[clusters == i]).any(axis=-1)
        ] = cluster_colors[i]
    clusters_statistics(
        image_compr.reshape(-1, 3), cluster_colors, clusters, "img_stat"
    )  # Very slow (:
    return (recolored * 255).astype(int)
