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

    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        return 1  # Empty matrix has determinant 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    return np.linalg.det(matrix)

def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues, _ = np.linalg.eig(matrix)

    if all(eigval > 0 for eigval in eigenvalues):
        return "Positive definite"
    elif all(eigval >= 0 for eigval in eigenvalues):
        return "Positive semi-definite"
    elif all(eigval < 0 for eigval in eigenvalues):
        return "Negative definite"
    elif all(eigval <= 0 for eigval in eigenvalues):
        return "Negative semi-definite"
    else:
        return "Indefinite"
