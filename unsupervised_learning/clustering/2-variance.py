#!/usr/bin/env python3
"""  that calculates the total intra-cluster variance for a data set """
import numpy as np


def variance(X, C):
    """

    X is a numpy.ndarray of shape (n, d)
    containing the data set
    C is a numpy.ndarray of shape (k, d)
      containing the centroid means for each cluster
    You are not allowed to use any loops
    Returns: var, or None on failure
        var is the total variance

    """
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None

    n, d = X.shape
    k, d = C.shape

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)**2
    nearest_centroid_indices = np.argmin(distances, axis=1)
    var = np.sum(np.min(distances, axis=1))

    return var
