import matplotlib.pyplot as plt

def visualize(x, y=None, y_hat=None):
    """
    Visualization helper function.
    """
    if y is not None:
        plt.plot(x, y, '-', label='Loss')
    if y and y_hat is not None:
        plt.plot(x, y, '0', label='Observed Values')
        plt.plot(x, y_hat, '-', label='Model Predictions')
    plt.legend()
    plt.show()
