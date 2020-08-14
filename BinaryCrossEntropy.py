import numpy as np

def BinaryCrossEntropyLoss(prediction, label, grad=False):
    if grad:
        if label == 0: ## if label is zero
            return 1/(1-prediction) ## compute deriv
        else: # else if label is one
            return -1/prediction ## compute deriv
    return -(label*np.exp(prediction)+(1-label)*np.exp(1-prediction))## compute binary cross entrpoy