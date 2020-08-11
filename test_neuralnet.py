import numpy as np
import neuralnet as nn


def test_activate_matrix_correct_output():
    np.testing.\
        assert_array_almost_equal( 
            x=nn.activate(
                precedent_layer=nn.layer(
                    input=np.empty([]),
                    output=np.array([[0.1, 0.2], [0.3, 0.4]]),
                    weight=np.empty([])),
                weight=np.array(
                    [[0.7, 0.8, 0.9], [0.1, 0.11, 0.12]])
            ).output,
            y=np.array(
                [[0.723122, 0.533699], [0.785835, 0.545127]]))


def test_activate_matrix_correct_input():
    np.testing.\
        assert_array_almost_equal(
            x=nn.activate(
                precedent_layer=nn.layer(
                    input=np.empty([]),
                    output=np.array([[0.1, 0.2], [0.3, 0.4]]),
                    weight=np.empty([])),
                weight=np.array(
                    [[0.7, 0.8, 0.9], [0.1, 0.11, 0.12]])
            ).input,
            y=np.array(
                [[0.96, 0.135], [1.3, 0.181]]))


def test_add_bias_units():
    np.testing.\
        assert_array_equal(
            x=nn.add_bias_units(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
            y=np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]]))


def test_J():
    np.testing.assert_almost_equal(
        actual=nn.J(
            h=np.array([
                [0.51457, 0.51113, 0.49746],
                [0.51459, 0.51114, 0.49745],
                [0.51456, 0.51107, 0.49739],
                [0.51452, 0.51098, 0.49735],
                [0.51449, 0.51096, 0.49735]]),
            y=np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]),
            m=5),
        desired=2.10095659)


def test_cost():
    np.testing.assert_almost_equal(
        actual=nn.cost(
            layer_one=nn.layer(
                input=np.empty([]),
                output=np.array([
                    [0.084147, -0.027942, -0.099999],
                    [0.090930, 0.065699, -0.053657],
                    [0.014112, 0.098936, 0.042017],
                    [-0.075680, 0.041212, 0.099061],
                    [-0.095892, -0.054402, 0.065029]]),
                weight=np.empty([])),
            layer_two_weight=np.array([
                [0.084147, -0.027942, -0.099999, -0.028790],
                [0.090930, 0.065699, -0.053657, -0.096140],
                [0.014112, 0.098936, 0.042017, -0.075099],
                [-0.075680, 0.041212, 0.099061, 0.014988],
                [-0.095892, -0.054402, 0.065029, 0.091295]]),
            layer_three_weight=np.array([
                [0.08414, -0.07568, 0.06569, -0.05440, 0.04201, -0.02879],
                [0.09093, -0.09589, 0.09893, -0.09999, 0.09906, -0.09614],
                [0.01411, -0.02794, 0.04121, -0.05365, 0.06502, -0.07509]]),
            y=np.array([[2], [3], [1], [2], [3]])),
        desired=2.10094675)
