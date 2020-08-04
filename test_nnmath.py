import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import nnmath as m

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

def test_nnmath_plot():
    x = np.linspace(-10, 10, 100)
    plt.plot( \
        x, \
        m.sigmoidgrad(x))    
    plt.plot( \
        x, \
        m.sigmoid(x))
    plt.show()