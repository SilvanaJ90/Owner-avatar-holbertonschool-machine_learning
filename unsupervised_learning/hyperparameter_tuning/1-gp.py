#!/usr/bin/env python3
""" Class GaussianProcess"""

import numpy as np


class GaussianProcess:
    """ Class GaussianProcess"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ that represents a noiseless 1D Gaussian process: """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ Calculates the covariance kernel matrix between two matrices """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) \
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
            that predicts the mean and standard deviation
            of points in a Gaussian process:
        """
        K_s = self.kernel(X_s, X_s)
        K = self.K
        K_inv = np.linalg.inv(K)
        K_s_X = self.kernel(X_s, self.X)
        mu = np.dot(K_s_X, np.dot(K_inv, self.Y)).flatten()
        sigma = np.sqrt(np.diag(K_s - np.dot(K_s_X, np.dot(K_inv, K_s_X.T))))
        return mu, sigma
