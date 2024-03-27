#!/usr/bin/env python3
""" calculates the marginal probability of obtaining the data:"""
intersection = __import__('1-intersection').intersection
import numpy as np


def marginal(x, n, P, Pr):
    """ Doc """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0: 
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1: 
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if (P < 0).any() or (P > 1).any():
        raise ValueError(
            "All values in P must be in the range [0, 1]")
    if (Pr < 0).any() or (Pr > 1).any():
        raise ValueError(
            "All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
