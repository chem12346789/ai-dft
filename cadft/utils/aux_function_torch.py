"""
Some useful functions for data preprocess.
"""

import torch


def sign_sqrt(a):
    """
    Compute the square root of the absolute value of a, with the sign of a.
    """
    return torch.abs(a) ** 0.5 * torch.sign(a)


def sign_square(a):
    """
    Compute the square of the absolute value of a, with the sign of a.
    """
    return torch.abs(a) ** 2 * torch.sign(a)
