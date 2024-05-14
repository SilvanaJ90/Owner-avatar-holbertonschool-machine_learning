#!/usr/bin/env python3
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ class BayesianOptimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ Bayesian optimization on a noiseless 1D Gaussian process: """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)
