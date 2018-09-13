"""
GRADIENT DESCENT FUNCTION
-------------------------------------------------------------------------------------
Updates weights from the derivative of the cost function

    Parameters
    ----------
    w0: scalar
        weight of intercepts
    w1: scalar
        weight of feature x1
    x1: nupmy array
        x-axis values
    y: numpy array
        y labels
    y_hat:numpy array
        predicted y values from model
    size: scalar
        number of training examples
    learning-rate: scalar
        learning rate alpha of algorithm (defaulted as 0.1)
"""

import numpy as np

def gradient_descent(w0,w1,x1,y,y_hat,size,learning_rate=0.1):
    diff = y_hat-y
    w0 = w0 - np.sum(diff)*(learning_rate/size)
    w1 = w1 - np.sum(diff*x1)*(learning_rate/size)
    return w0, w1
