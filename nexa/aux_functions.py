"""
Auxiliary functions for the nexa machinery
"""

import numpy as np


def softmax_base(vector):
    """
    This is the softmax base to test other functions
    that do take matrices. This only takes a vector
    and returns the softmax version of it. That, is
    normalized to one through the use of an exponential.

    Parameters:
    -----------
    vector : a vector of numbers

    Return:
    A list of the  same length as vector with non-negative
    numbers
    """
    # Calculate the maximum
    max_value = np.max(vector)
    soft = np.zeros(vector.shape)
    division = np.sum(np.exp(vector - max_value))

    for index, x in enumerate(vector):
        numerator = np.exp(x - max_value)
        result = numerator / division
        soft[index] = result

    return soft
