from collections import namedtuple
import nnmath as m
import numpy as np


def activate(x, theta):
    activation = namedtuple("activation", ["input", "output"])
    input = x @ np.transpose(theta)
    return activation(
        input=input,
        output=m.sigmoid(input))


def add_bias_units(A):
    return np.c_[np.ones(A.shape[0]), A]
