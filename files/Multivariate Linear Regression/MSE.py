"""
LOSS FUNCTION
-------------------------------------------------------------------------------------
Calculate the mean squared error for given labels and model outputs

    Parameters
    ----------
    y: n x 1 size array
        y labels
    y_hat: n x 1 size array
        predicted y values from model
"""

import numpy as np

def calculate_MSE(y, y_hat):
    size=len(y_hat)
    diff= y_hat - y
    diff_squared=np.power(diff,2)
    loss=1/(2*size)*np.sum(diff_squared)
    return loss
