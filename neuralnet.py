from collections import namedtuple
import nnmath as nnm
import numpy as np
import scipy as sp
from sklearn.preprocessing import OneHotEncoder

layer = namedtuple("layer", ["input", "output", "weight"])
iteration = namedtuple("iteration", [
    "J",
    "layer_two_weight_grad",
    "layer_three_weight_grad"])


def learn(
    weights_vector,
    weight_one_size,
    X,
    y
):
    return to_matrices(
        sp.optimize.minimize(
            fun=run_weights_vectorized,
            x0=weights_vector,
            args=(
                weight_one_size,
                X,
                y),
            method='Newton-CG',
            jac=True,
            options={'maxiter': 150}).x,
        matrix_one_rows=weight_one_size,
        matrix_one_columns=X.shape[1] + 1,
        matrix_two_rows=np.unique(y).shape[0],
        matrix_two_columns=weight_one_size + 1)


def run_weights_vectorized(
    weights_vector,
    weight_one_size,
    X,
    y
):
    weights = to_matrices(
        weights_vector,
        matrix_one_rows=weight_one_size,
        matrix_one_columns=X.shape[1] + 1,
        matrix_two_rows=np.unique(y).shape[0],
        matrix_two_columns=weight_one_size + 1)
    iteration = run(
        layer_one=layer(
            input=np.empty([]),
            output=X,
            weight=np.empty([])),
        layer_two_weight=weights[0],
        layer_three_weight=weights[1],
        y=y)
    return (
        iteration.J,
        to_vector(
            iteration.layer_two_weight_grad,
            iteration.layer_three_weight_grad))


def run(
    layer_one,
    layer_two_weight,
    layer_three_weight,
    y
):
    layer_two = activate(
        layer_one,
        layer_two_weight)
    layer_three = activate(
        layer_two,
        layer_three_weight)
    m = layer_one.output.shape[0]
    Y = OneHotEncoder(sparse=False).fit_transform(y)
    delta3 = layer_three.output - Y
    Delta2 = np.transpose(delta3) @ add_bias_units(layer_two.output)
    delta2 = \
        (delta3 @ remove_bias_units(layer_three_weight)) \
        * nnm.sigmoid_grad(layer_two.input)
    Delta1 = np.transpose(delta2) @ add_bias_units(layer_one.output)
    return iteration(
        J=cost(
            h=layer_three.output,
            Y=Y,
            m=m),
        layer_two_weight_grad=Delta1/m,
        layer_three_weight_grad=Delta2/m)


def cost(h, Y, m):
    return \
        (np.sum(
            np.sum(
                (-Y * np.log(h))
                - ((1-Y) * (np.log(1-h)))))) \
        / m


def activate(precedent_layer, weight):
    input = add_bias_units(precedent_layer.output) @ np.transpose(weight)
    return layer(
        input=input,
        output=nnm.sigmoid(input),
        weight=weight)


def add_bias_units(matrix):
    return np.c_[np.ones(matrix.shape[0]), matrix]


def remove_bias_units(matrix):
    return matrix[:, 1:]


def to_vector(matrix_one, matrix_two):
    return np.concatenate((np.ravel(matrix_one), np.ravel(matrix_two)))


def to_matrices(
    array,
    matrix_one_rows,
    matrix_one_columns,
    matrix_two_rows,
    matrix_two_columns
):
    split_index = matrix_one_rows * matrix_one_columns
    return (
        np.reshape(
            array[0:split_index],
            (matrix_one_rows, matrix_one_columns)),
        np.reshape(
            array[split_index:array.shape[0]],
            (matrix_two_rows, matrix_two_columns)))
