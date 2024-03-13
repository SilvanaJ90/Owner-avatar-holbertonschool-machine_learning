#!/usr/bin/env python3
""" Doc """


class Exponential:
    """ Class Exponential """
    def __init__(self, data=None, lambtha=1.):
        e = 2.7182818285
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """ Calculates the value of the PMF """
        e = 2.7182818285
        x = float(x)
        if x < 0:
            return 0
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """ Calculates the value of the CDF """
        e = 2.7182818285
        if x < 0:
            return 0
        return (1 - e**(-self.lambtha*x))
