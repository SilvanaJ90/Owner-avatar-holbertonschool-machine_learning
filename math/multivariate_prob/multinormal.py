#!/usr/bin/env python3
""" that represents a Multivariate Normal distribution: """
import numpy as np


class MultiNormal:
    """ Doc """
    def __init__(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        
        self.mean = np.mean(data, axis=1, keepdims=True)
        centered_data = data - self.mean
        self.cov = np.dot(centered_data, centered_data.T) / (data.shape[1] - 1)
        
    def pdf(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.ndim != 2 or x.shape != (self.mean.shape[0], 1):
            raise ValueError(f"x must have the shape ({self.mean.shape[0]}, 1)")
        
        d = self.mean.shape[0]
        exponent = -0.5 * np.dot(np.dot((x - self.mean).T, np.linalg.inv(self.cov)), (x - self.mean))
        coefficient = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(self.cov)))
        pdf_value = coefficient * np.exp(exponent)
        
        return pdf_value
