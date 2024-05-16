"""
Some useful functions for data preprocess.
"""

import numpy as np


def sign_sqrt(a):
    """
    Compute the square root of the absolute value of a, with the sign of a.
    """
    return np.abs(a) ** 0.5 * np.sign(a)


def sign_square(a):
    """
    Compute the square of the absolute value of a, with the sign of a.
    """
    return np.abs(a) ** 2 * np.sign(a)
