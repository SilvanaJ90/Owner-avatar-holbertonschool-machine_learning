#!/usr/bin/env python3
""" Doc """
import numpy as np

def initialize(X, k):
    """
    Initialize cluster centroids  K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids each cluster,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    centroids = np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))

    return centroids

def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing the centroid means each cluster
        clss: numpy.ndarray of shape (n,) containing the index of the cluster each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    n, d = X.shape

    if n == 0:
        return None, None

    # Convert iterations to integer if it's a string
    try:
        iterations = int(iterations)
    except ValueError:
        print("Warning: iterations parameter should be an integer. Defaulting to 1000.")
        iterations = 1000

    # Initialize centroids
    C = initialize(X, k)
    clss = np.zeros(n)

    for _ in range(iterations):
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((X[:, np.newaxis, :] - C) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=1)

        # Update centroids
        new_C = np.array([X[clss == i].mean(axis=0) if np.sum(clss == i) > 0 else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0)) for i in range(k)])

        if np.array_equal(C, new_C):
            break

        C = new_C

    return C, clss