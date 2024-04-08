#!/usr/bin/env python3
""" Doc """
import numpy as np


def initialize(X, k):
    """
    that initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or not isinstance(k, int) or k <= 0:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    centroids = np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))

    return centroids
