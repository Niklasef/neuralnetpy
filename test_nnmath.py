import numpy as np
import nnmath as m
from numpy.testing import assert_allclose

def test_sigmoid_scalar():
    assert m.sigmoid(5) == 0.9933071490757153

def test_sigmoid_vector():
    assert_allclose( \
        m.sigmoid(np.array([0.001, 0.002])), \
        np.array([0.50025, 0.50050]))

def test_sigmoid_matrix():
    assert_allclose( \
        m.sigmoid(np.array([[0.001, 0.002], [0.002, 0.003]])), \
        np.array([[0.50025, 0.50050], [0.50050, 0.50075]]))
