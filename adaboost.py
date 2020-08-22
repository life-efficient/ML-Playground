from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append('..') # add utils file to path
from utils import get_classification_data, calc_accuracy, visualise_predictions, show_data
import numpy as np
import matplotlib.pyplot as plt
import json



class AdaBoost:
    def __init__(self, n_models):
        self.n_models = n_models
        self.models = []

    def code_labels(self, Y):
        Y[Y==0] = -1
        return Y

    def resample(self, X, Y, weights):
        # print(weights)
        idxs = np.random.choice(range(len(X)), size=len(X), replace=True, p=weights)
        X = X[idxs]
        Y = Y[idxs]
        return X, Y
    
    def compute_model_error(self, predictions, labels, weights):
        return np.sum(weights[labels != predictions]) / np.sum(weights)

    def compute_model_alpha(self, error):
        return np.log((1 - error) / ( error + 0.01) ) / 2

    def update_weights(self, weights, alpha, predictions, labels):
        weights = weights * np.exp(-alpha * predictions * labels) 
        weights /= np.sum(weights)
        return weights

    def fit(self, X, Y):
        Y = self.code_labels(Y)
        example_weights = np.ones(len(X)) / len(X)
        # print(example_weights)
        for _ in range(self.n_models):
            bootstrapped_X, bootstrapped_Y = self.resample(X, Y, example_weights)
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(bootstrapped_X, bootstrapped_Y)
            predictions = model.predict(X)
            error = self.compute_model_error(predictions, Y, example_weights)
            alpha = self.compute_model_alpha(error)
            model.weight = alpha
            self.models.append(model)
            example_weights = self.update_weights(example_weights, alpha, predictions,Y)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.weight * model.predict(X)
        # predictions = np.sign(predictions)
        return predictions

    def __repr__(self):
        return json.dumps([
            m.weight for m in self.models
        ])

    def calc_accuracy(self, X, Y):
        print( np.mean(self.predict(X) == Y))

X, Y = get_classification_data(m=100, sd=4)
show_data(X, Y)

adaboost = AdaBoost(n_models=10)
adaboost.fit(X, Y)
print(adaboost)
adaboost.calc_accuracy(X, Y)
visualise_predictions(adaboost.predict, X, Y)