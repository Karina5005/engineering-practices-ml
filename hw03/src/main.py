from sklearn.datasets import (
    make_blobs,
    make_moons,
)

from agglomertive import AgglomertiveClustering
from kmeans import KMeans
from utils import (
    clusterize_image,
    read_image,
    save_image,
    visualize_clasters,
)

if __name__ == "__main__":
    X_1, true_labels = make_blobs(
        400, 2, centers=[[0, 0], [-4, 0], [3.5, 3.5], [3.5, -2.0]]
    )
    visualize_clasters(X_1, true_labels, "claster1")
    X_2, true_labels = make_moons(400, noise=0.075)
    visualize_clasters(X_2, true_labels, "claster2")

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_1)
    labels = kmeans.predict(X_1)
    visualize_clasters(X_1, labels, "kmeans1")

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_2)
    labels = kmeans.predict(X_2)
    visualize_clasters(X_2, labels, "kmeans2")

    agg_clustering = AgglomertiveClustering(n_clusters=4, linkage="complete")
    labels = agg_clustering.fit_predict(X_1)
    visualize_clasters(X_1, labels, "agg1")

    agg_clustering = AgglomertiveClustering(n_clusters=2, linkage="single")
    labels = agg_clustering.fit_predict(X_2)
    visualize_clasters(X_2, labels, "agg2")

    image = read_image("data/Lena.png")
    result = clusterize_image(image)
    save_image(result, "result.png")
