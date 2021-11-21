import numpy as np


def compute_scale_down(input_size, output_size):
    """вычисляем масштаб сжатия нейронной сети."""
    return output_size[0] / input_size[0]


def prob_true(p):
    """True p"""
    return np.random.random() < p
