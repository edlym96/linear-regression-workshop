"""
MODEL FUNCTION
-------------------------------------------------------------------------------------
Calculate the linear regression output for given feature, x1 and weights, w0 and w1

    Parameters
    ----------
    x1: numpy array
        x-axis values
    w0: scalar
        weight for intecept terms
    w1: scalar
        weight for feature x1

"""


def simple_linear_model(w0,w1,x1):
    y_hat=w0+x1*w1
    return y_hat