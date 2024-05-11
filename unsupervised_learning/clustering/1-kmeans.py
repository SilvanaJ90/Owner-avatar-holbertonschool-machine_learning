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

    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))

    if centroids is None:
        return None, None

    for i in range(iterations):
        # Asignación de Puntos a los Centroides más Cercanos: dist(p,q)=i=1∑d​(pi​−qi​)2
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        C = np.array([X[clss == c].mean(axis=0) for c in range(k)])

        C = np.array([X[clss == c].mean(axis=0) if np.sum(clss == c) > 0 else np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0)) for c in range(k)])
        
        for c in range(k):
            if X[clss == c].size == 0:
                C[c] = np.random.uniform(min_vals, max_vals, size=(1, d))


        if np.array_equal(centroids, C):
            break


    return centroids, clss