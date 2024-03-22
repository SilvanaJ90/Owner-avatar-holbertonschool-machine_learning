#!/usr/bin/env python3
"""   that calculates the definiteness of a matrix:: """
cofactor = __import__('2-cofactor').cofactor
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a given matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        the definiteness matrix of matrix
    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.

    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues, _ = np.linalg.eig(matrix)
    num_pos = np.sum(eigenvalues > 0)
    num_neg = np.sum(eigenvalues < 0)
    num_zero = np.sum(np.isclose(eigenvalues, 0))

    if num_pos == matrix.shape[0]:
        return "Positive definite"
    elif num_pos > 0 and num_zero > 0:
        return "Positive semi-definite"
    elif num_neg == matrix.shape[0]:
        return "Negative definite"
    elif num_neg > 0 and num_zero > 0:
        return "Negative semi-definite"
    else:
        return "Indefinite"