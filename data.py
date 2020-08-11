import numpy as np

def get_regression_data(m=20): 
    ground_truth_w = 2.3 # slope
    ground_truth_b = -8 #intercept
    X = np.random.uniform(0, 1, size=(m, 1))*2
    idxs = np.argsort(X, axis=0)
    idxs = np.squeeze(idxs)
    X = X[idxs]
    Y = ground_truth_w*X + ground_truth_b + 0.2*np.random.randn(m, 1)
    return X, Y