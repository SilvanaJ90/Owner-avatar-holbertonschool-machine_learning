#!/usr/bin/env python3
""" that represents a Multivariate Normal distribution: """
import numpy as np


class MultiNormal:
    """ Doc """
    def __init__(self, data):
        """Doc """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1).reshape(-1, 1)
        self.cov = np.dot(
            data - self.mean, (data - self.mean).T) / (data.shape[1] - 1)
