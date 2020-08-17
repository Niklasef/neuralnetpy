from collections import namedtuple
import nnmath as nnm
import numpy as np
import scipy as sp
from sklearn.preprocessing import OneHotEncoder

layer = namedtuple("layer", ["input", "output", "weight"])
cost_result = namedtuple("cost_result", [
    "J",
    "layer_two_weight_grad",
    "layer_three_weight_grad"])


def minimize(
    params,
    matrix_one_rows,
    matrix_one_columns,
    matrix_two_rows,
    matrix_two_columns,
    X,
    y
):
    return sp.optimize.minimize(
        fun=run,
        x0=params,
        args=(
            matrix_one_rows,
            matrix_one_columns,
            matrix_two_rows,
            matrix_two_columns,
            X,
            y),
        method='Newton-CG',
        jac=True,
        options={'maxiter': 150}).x


def run(
    params,
    matrix_one_rows,
    matrix_one_columns,
    matrix_two_rows,
    matrix_two_columns,
    X,
    y
):
    weights = to_matrices(
        params,
        matrix_one_rows,
        matrix_one_columns,
        matrix_two_rows,
        matrix_two_columns)
    cost_result = cost(
        layer_one=layer(
            input=np.empty([]),
            output=X,
            weight=np.empty([])),
        layer_two_weight=weights[0],
        layer_three_weight=weights[1],
        y=y)
    return cost_result.J, to_vector(
        cost_result.layer_two_weight_grad,
        cost_result.layer_three_weight_grad)


def cost(layer_one, layer_two_weight, layer_three_weight, y):
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
    return cost_result(
        J=J(
            h=layer_three.output,
            Y=Y,
            m=m),
        layer_two_weight_grad=Delta1/m,
        layer_three_weight_grad=Delta2/m)


def J(h, Y, m):
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
