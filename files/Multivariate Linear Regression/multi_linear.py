"""
MODEL FUNCTION
-------------------------------------------------------------------------------------
Calculate the linear regression output for given feature, x1 and weights, w0 and w1

    Parameters
    ----------
    X: m x n size array
        feature values
    W: n x 1 size array
        weight values

"""
import numpy as np

def multi_linear_model(W,X):
    y_hat=np.dot(X,W)
    return y_hat