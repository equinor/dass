"""Tapering function for use in localisation.
"""

import numpy as np


def gauss(distances, radius):
    return np.exp(-0.5 * (distances / radius) ** 2)
