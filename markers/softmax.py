import numpy as np


def softmax(x1, x2):
    e_x1 = np.exp(x1)
    e_x2 = np.exp(x2)

    return (e_x1 / (e_x1 + e_x2)) * (e_x2 / (e_x1 + e_x2))


def calc_value(x1, x2):
    return np.prod(softmax(x1, x2))
