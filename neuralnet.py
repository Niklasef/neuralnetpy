import nnmath as m
import numpy as np


def hypothesis(x, theta):
    return m.sigmoid(x @ np.transpose(theta))
