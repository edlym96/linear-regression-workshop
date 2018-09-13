"""
NORMALISATION FUNCTION
-------------------------------------------------------------------------------------
Normalise weights by subtracting each weight by the mean and dividing by the standard deviation

    Parameters
    ----------
    X: m x n size array
        feature values
"""

import numpy as np

def normalise(X):
    mean_x=np.mean(X,axis=0)
    std_x=np.std(X,axis=0)
    normalised_x=(X-mean_x)/std_x
    return normalised_x
