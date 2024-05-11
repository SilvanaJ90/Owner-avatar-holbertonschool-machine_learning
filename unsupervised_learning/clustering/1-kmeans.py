#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means clustering.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset for clustering
        k: positive integer, number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids for each cluster,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or not isinstance(k, int) or k <= 0:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1]))

    return centroids


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
        clss: numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    centroids = initialize(X, k)
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
