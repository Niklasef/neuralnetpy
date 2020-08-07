import numpy as np
import neuralnet as nn

def test_hypothesis_matrix():
    np.testing.assert_array_almost_equal( \
        x=nn.hypothesis(\
            x=np.array(\
                [[0.1, 0.2, 0.3],\
                [0.4, 0.5, 0.6]]),\
            theta=np.array(\
                [[0.7, 0.8, 0.9],\
                [0.1, 0.11, 0.12]])), \
        y=np.array(\
            [[0.622459, 0.516993],\
            [0.772064, 0.541653]]))