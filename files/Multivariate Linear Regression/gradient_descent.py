"""
GRADIENT DESCENT FUNCTION
-------------------------------------------------------------------------------------
Updates weights from the derivative of the cost function

    Parameters
    ----------
    W: n x 1 size array
        weight values
    X: m x n size array
        feature values
    y: m x 1 size array
        y labels
    y_hat:n x 1 size array
        predicted y values from model
    size: scalar
        number of training examples
    learning-rate: scalar
        learning rate alpha of algorithm (defaulted as 0.1)
"""

import numpy as np

def gradient_descent(W,X,y,y_hat,size,learning_rate=0.01):
    diff = y_hat-y
    W = W-(learning_rate/size)*np.dot(np.transpose(X),diff)
    print("W is: ", W)
    return W
