from collections import namedtuple
import nnmath as m
import numpy as np

layer = namedtuple("activation", ["input", "output", "weight"])


def activate(precedent_layer, weight):
    input = precedent_layer.output @ np.transpose(weight)
    return layer(
        input=input,
        output=m.sigmoid(input),
        weight=weight)


def add_bias_units(matrix):
    return np.c_[np.ones(matrix.shape[0]), matrix]
