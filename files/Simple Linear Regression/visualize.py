import matplotlib.pyplot as plt

def visualize(x, y=None, y_hat=None):
    """Visualization helper function.

    Parameters
    ----------
    x: array-like
        x-axis values
    y: array-like
        y-axis target values
    y_noise: array-like
        y-axis observations
    y_hat: array-like
        y-axis model predictions
    """
    if y is not None:
        plt.plot(x, y, 'o', label='Observed Values')
    if y_hat is not None:
        plt.plot(x, y_hat, '-', label='Model Predictions')
    plt.legend();
    plt.show()