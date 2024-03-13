#!/usr/bin/env python3
"""
    Calculates the shape of a matrix.
    Args: matrix: a nested list representing a matrix
    Returns: list
"""


def matrix_shape(matrix):
    """ Calculates the shape of a matrix. """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
