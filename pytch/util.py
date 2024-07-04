import logging
import numpy as np

logger = logging.getLogger("pytch.util")


def f2cent(f, standard_frequency):
    return 1200.0 * np.log2((np.maximum(f, 0.0) + 0.1) / standard_frequency)


def cent2f(p, standard_frequency):
    return np.exp2(p / 1200.0) * standard_frequency - 0.1


def consecutive(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def index_gradient_filter(x, y, max_gradient):
    """Get index where the abs gradient of x, y is < max_gradient."""
    return np.where(np.abs(np.diff(y) / np.diff(x)) < max_gradient)[0]
