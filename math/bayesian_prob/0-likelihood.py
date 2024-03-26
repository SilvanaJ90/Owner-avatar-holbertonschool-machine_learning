#!/usr/bin/env python3
""" that calculates the likelihood of obtaining this data given various
 hypothetical probabilities of developing severe side effects:
"""


def likelihood(x, n, P):
    """
    Arg:
        -x is the number of patients that develop severe side effects
        - n is the total number of patients observed
        - P is a 1D numpy.ndarray containing the various
         hypothetical probabilities of developing severe side effects
    -Conditions:
        - If n is not a positive integer, raise a ValueError
         with the message n must be a positive integer
        - If x is not an integer that is greater than or 
        equal to 0, raise a ValueError with the message x 
        must be an integer that is greater than or equal to 0
        - If x is greater than n, raise a ValueError with the
         message x cannot be greater than n
        - If P is not a 1D numpy.ndarray, raise a TypeError
         with the message P must be a 1D numpy.ndarray
        - If any value in P is not in the range [0, 1], raise a 
        ValueError with the message All values in P must be in the range [0, 1]
    Returns: a 1D numpy.ndarray containing the likelihood of obtaining the data,
    x and n, for each probability in P, respectively
    """
