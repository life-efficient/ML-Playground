import matplotlib.pyplot as plt
import numpy as np

def visualise_regression_data(X, Y, H=None):
    print(X)
    ordered_idxs = np.argsort(X, axis=0)
    print(ordered_idxs)
    X = X[ordered_idxs]
    Y = Y[ordered_idxs]
    plt.figure()
    print(X)
    print(X.shape)
    plt.scatter(X, Y, c='r', label='Label')
    if H is not None:
        domain = np.linspace(min(X), max(X))
        y_hat = H(domain)
        plt.plot(domain, y_hat, label='Hypothesis')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()