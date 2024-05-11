#!/usr/bin/env python3
""" Doc """
import numpy as np


def kmeans(X, k, iterations=1000):
    """ Doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape

    # Inicializar los centroides de los clusters
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))

    if centroids is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        C = np.array([X[clss == c].mean(axis=0) if np.sum(clss == c) > 0 else np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0)) for c in range(k)])

        if np.array_equal(centroids, C):
            break

        centroids = C

    return centroids, clss