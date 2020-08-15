import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import sklearn.datasets
import sys
sys.path.append('..')
from utils import get_classification_data, show_data, visualise_predictions, colors

m = 60
n_features = 2
n_classes = 2
X, Y = get_classification_data(sd=10, m=60, n_clusters=n_classes, n_features=n_features)
show_data(X, Y)



class classificationTree:
    def __init__(self,max_depth):
        self.max_depth = max_depth

    def predict(self, X, y):
        # run fit until impurtiy = 0
        best_feature, best_splitpoint = fit(X, y)
        
       
        def fit(X, y):
            X = np.sort(X) # sort the data
            best_feature, best_splitpoint = find_best_split(X, y)
            return best_feature, best_splitpoint
        
        def find_best_split(X, y):
            best_feature_idx = 0
            best_impurity = 1
            splitPoint = 0
            for feature_idx, examples in enumerate(X.T): # loop through each feature
                current_impurity, current_splitpoint = find_best_split_on_feature(examples, y)
                if currentImpurity < best_impurity:
                    impurity = current_impurity
                    best_feature_idx = feature_idx
                    splitPoint = current_splitpoint
            return best_feature_idx, splitPoint
        
        def find_best_split_on_feature(x, y): # takes in a list of examples for one feature
            best_impurity = 1
            best_split_point = 0
            for i in range(len(x) - 1): # loop through Xs until X-1
                splitPoint = np.mean(x[i], x[i+1])
                currentImpurity = getImpurity(splitPoint, x, y)
                if currentImpurity < best_impurity:
                    best_impurity = currentImpurity
                    best_split_point = splitPoint
            return best_impurity, best_splitPoint

        
        def getImpurity(split_point, x, y): # get the impurity
            left_labels = y[x < split_point]
            right_labels = y[x > split_point]
            p_squared = 0
            # run gini impurity for left side 
            for _class in np.unique(y): # gini_impurity_per_region()
                class_count = left_labels.count(_class) 
                p_squared += (class_count/len(left_labels))**2
            left_side_impurity = 1 - p_squared

            # run gini impurity for right side
            p_squared = 0
            for _class in np.unique(y): # gini_impurity_per_region()
                class_count = right_labels.count(_class)
                p_squared += (class_count/len(right_labels))**2
            right_side_impurity = 1 - p_squared

            # return total gini impurity
            proportion_left = len(left_labels) / len(x)
            proportion_right = len(right_labels) / len(x)
            impurity = proportion_left * left_side_impurity + proportion_right * right_side_impurity
            return impurity 
            
        return best_feature, best_splitpoint

model = classificationTree()
print(model(predict(X,Y)))

    
    


    