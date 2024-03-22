#!/usr/bin/env python3
"""  that calculates the adjugate matrix of a matrix: """
cofactor = __import__('2-cofactor').cofactor
import numpy as np


def adjugate(matrix):
    """
    Calculates the adjugate of a given matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        the adjugate matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    """
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = np.array(cofactor_matrix)   

    return adjugate_matrix.T
