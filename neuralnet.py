from collections import namedtuple
import nnmath as m
import numpy as np
from sklearn.preprocessing import OneHotEncoder

layer = namedtuple("layer", ["input", "output", "weight"])


def activate(precedent_layer, weight):
    input = add_bias_units(precedent_layer.output) @ np.transpose(weight)
    return layer(
        input=input,
        output=m.sigmoid(input),
        weight=weight)


def add_bias_units(matrix):
    return np.c_[np.ones(matrix.shape[0]), matrix]
