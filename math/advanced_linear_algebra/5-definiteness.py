#!/usr/bin/env python3
"""   that calculates the definiteness of a matrix:: """
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

    if matrix.ndim < 2:
        matrix = np.atleast_2d(matrix)

    if not np.array_equal(matrix, matrix.T):
        return None  # Matriz no es simétrica

    eigenvalues, _ = np.linalg.eig(matrix)

    positive_eigenvalues = np.sum(eigenvalues > 0)
    zero_eigenvalues = np.sum(eigenvalues == 0)

    if positive_eigenvalues == matrix.shape[0]:
        return "Positive definite"
    elif positive_eigenvalues + zero_eigenvalues == matrix.shape[0]:
        return "Positive semi-definite"
    elif positive_eigenvalues == 0:
        return "Negative definite"
    elif positive_eigenvalues == 1 and np.all(np.isclose(eigenvalues, eigenvalues[0])):
        return "Indefinite"
    elif positive_eigenvalues <= 1 and np.all(np.isclose(eigenvalues, eigenvalues[-1])):
        return "Negative semi-definite"
    else:
        return None