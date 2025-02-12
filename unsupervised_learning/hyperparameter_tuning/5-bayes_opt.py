#!/usr/bin/env python3
""" Bayesian optimization on a noiseless 1D Gaussian process: """

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ class BayesianOptimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing the
            inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the
            outputs of the black-box function for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds of the
            space in which to look for the optimal point
        ac_samples is the number of samples that
            should be analyzed during acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to
            the output of the black-box function
        xsi is the exploration-exploitation factor for acquisition
        minimize is a bool determining whether optimization should be performed
            for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)

    def acquisition(self):
        """
        that calculates the next best sample location:
            Uses the Expected Improvement acquisition function
            Returns: X_next, EI
                X_next is a numpy.ndarray of shape (1,)
                representing the next best sample point
                EI is a numpy.ndarray of shape (ac_samples,)
                containing the expected improvement of each potential sample
        You may use from scipy.stats import norm
        """
        y_pred, y_std = self.gp.predict(self.X_s)

        if self.minimize:
            best_idx = np.argmin(self.gp.Y)
            best_y = self.gp.Y[best_idx]

            z = (best_y - y_pred - self.xsi) / y_std
            EI = (best_y - y_pred - self.xsi) * norm.cdf(z) + \
                y_std * norm.pdf(z)
        else:
            best_idx = np.argmax(self.gp.Y)
            best_y = self.gp.Y[best_idx]

            z = (best_y - y_pred - self.xsi) / y_std
            EI = (best_y - y_pred - self.xsi) * norm.cdf(z) + \
                y_std * norm.pdf(z)

        x_next = self.X_s[np.argmax(EI)]
        return x_next, EI

    def optimize(self, iterations=100):
        """
        That optimizes the black-box function:
            iterations is the maximum number of iterations to perform
            If the next proposed point is one that has already been sampled,
                optimization should be stopped early
            Returns: X_opt, Y_opt
        X_opt is a numpy.ndarray of shape (1,) representing the optimal point
        Y_opt is a numpy.ndarray of shape (1,) representing
            the optimal function value
        """
        for i in range(iterations):
            x_next, EI = self.acquisition()
            # check if x has already been sampled
            if np.any(np.isclose(x_next, self.gp.X)):
                break
            y_next = self.f(x_next)
            self.gp.update(x_next, y_next)

        # get optimal point and value
        best_idx = np.argmin(self.gp.Y) if self.minimize \
            else np.argmax(self.gp.Y)
        self.gp.X = self.gp.X[:-1, :]
        x_opt = self.gp.X[best_idx]
        y_opt = self.gp.Y[best_idx]
        return x_opt, y_opt
