#!/usr/bin/env python3
"""calculates the determinant of a matrix: """


def determinant(matrix):
    """
        matrix is a list of lists whose
        determinant should be calculated
        If matrix is not a list of lists,
        raise a TypeError with the message matrix must be a list of lists
        If matrix is not square, raise a ValueError
        with the message matrix must be a square matrix
        The list [[]] represents a 0x0 matrix
        Returns: the determinant of matrix

        | a  b |
        | c  d |

        det = (a * d) - (b * c)

    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) > 0:
        for i in matrix:
            if not isinstance(i, list):
                raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in matrix:
        if len(i) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    num_rows = len(matrix)
    # Base case: 0x0 matrix
    if num_rows == 0:
        return 1

    # Base case: 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if num_rows == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det =  (a * d) - (b * c)
        
        return det
