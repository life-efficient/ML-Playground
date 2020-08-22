import matplotlib.pyplot as plt
import numpy as np

colors = [
 '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'
]

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

def show_data(X, Y, predictions=None):
    for i in range(min(Y), max(Y)+1):
        y = Y == i
        x = X[y]
        plt.scatter(x[:, 0], x[:, 1], c=colors[i])
        if predictions is not None:
            y = predictions == i
            x = X[y]
            plt.scatter(x[:, 0], x[:, 1], c=colors[i], marker='x', s=100)
    plt.show()