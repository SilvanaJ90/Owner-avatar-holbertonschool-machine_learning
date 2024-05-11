#!/usr/bin/env python3
""" Tests for the optimum number of clusters by variance: """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number
      of clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number
      of clusters to check for (inclusive)
    iterations is a positive integer containing the
    maximum number of iterations for K-means
    This function should analyze at least 2 different cluster sizes
    You should use:
        kmeans = __import__('1-kmeans').kmeans
        variance = __import__('2-variance').variance
    You may use at most 2 loops
    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs of K-means
          for each cluster size
        d_vars is a list containing the difference in variance
        from the smallest cluster size for each cluster size

    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax < 1:
        return None, None
    if kmax <= kmin:
        return None, None
    if type(StopIteration) is not int or iterations < 1:
        return None, None
    results = []
    var = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var.append(variance(X, C))
    d0 = var[0]
    d_vars = []
    for v in var:
        d_vars.append(d0 - v)
        return results, d_vars
