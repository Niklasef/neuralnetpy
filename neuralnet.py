from collections import namedtuple
import nnmath as m
import numpy as np
from sklearn.preprocessing import OneHotEncoder

layer = namedtuple("layer", ["input", "output", "weight"])


def cost(layer_one, layer_two_weight, layer_three_weight, y):
    layer_two = activate(
        layer_one,
        layer_two_weight)
    layer_three = activate(
        layer_two,
        layer_three_weight)
    return J(
        h=layer_three.output,
        y=OneHotEncoder(sparse=False).fit_transform(y),
        m=layer_one.output.shape[0])


def J(h, y, m):
    return \
        (np.sum(
            np.sum(
                (-y * np.log(h))
                - ((1-y) * (np.log(1-h)))))) \
        / m


def activate(precedent_layer, weight):
    input = add_bias_units(precedent_layer.output) @ np.transpose(weight)
    return layer(
        input=input,
        output=m.sigmoid(input),
        weight=weight)


def add_bias_units(matrix):
    return np.c_[np.ones(matrix.shape[0]), matrix]
