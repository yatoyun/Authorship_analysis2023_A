import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calc_value(x1, x2):
    return np.prod(softmax(np.array([x1, x2])))
