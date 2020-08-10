import numpy as np
import neuralnet as nn


def test_activate_matrix_correct_output():
    np.testing.\
        assert_array_almost_equal( 
            x=nn.activate(
                precedent_layer=nn.layer(
                    input=np.empty([]),
                    output=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                    weight=np.empty([])),
                weight=np.array(
                    [[0.7, 0.8, 0.9], [0.1, 0.11, 0.12]])
            ).output,
            y=np.array(
                [[0.622459, 0.516993], [0.772064, 0.541653]]))


def test_activate_matrix_correct_input():
    np.testing.\
        assert_array_almost_equal(
            x=nn.activate(
                precedent_layer=nn.layer(
                    input=np.empty([]),
                    output=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                    weight=np.empty([])),
                weight=np.array(
                    [[0.7, 0.8, 0.9], [0.1, 0.11, 0.12]])
            ).input,
            y=np.array(
                [[0.5, 0.068], [1.22, 0.167]]))


def test_add_bias_units():
    np.testing.\
        assert_array_equal(
            x=nn.add_bias_units(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
            y=np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]]))
