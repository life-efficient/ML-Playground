import numpy as np


class RandomTree:
    def __init__(self, max_depth=1, decision_tree=None):
        
        self.decision_tree = decision_tree
        self.max_depth = max_depth
    

    def branch_read(self, branch, X):
        left_selector = X[:, branch['feature']] < branch['splitpoint']

        right_selector = X[:, branch['feature']] > branch['splitpoint']

        if type(branch['left']) == dict:
            left = self.branch_read(branch['left'], X[left_selector])
        else: 
            left = np.array([branch['left']]*sum(left_selector))

        if type(branch['right']) == dict:
            right = self.branch_read(branch['right'], X[right_selector])
        else: 
            right = np.array([branch['right']]*sum(right_selector))

        return np.concatenate((left, right))


    def __call__(self, X):
        return self.branch_read(self.decision_tree, X)
        
        
        # return predictions

    
    def splitter(self, X, y):

        def find_split_points(x):
            x.sort()
            split_points = np.array([np.mean((x[idx], x[idx+1])) for idx in range(len(x)-1) ])
            # for idx in range(len(X)-1):
            #     split_points.append( np.mean(X[idx], X[idx+1])  )
            return split_points

        
        
        def gini_impurity(x, y, split_point): #calculate gini impurity
            right_split = x > split_point
            right_y = y[right_split]
            p_right = 1 - sum([np.mean(right_y == _class) **2 for _class in set(right_y)])
            # for _class in set(right_y):
            #     np.mean(right_y == _class) **2
            left_split = x < split_point
            left_y = y[left_split]
            p_left = 1 - sum([np.mean(left_y == _class) **2 for _class in set(left_y)])
            
            p_total = (p_left*len(left_y) + p_right*len(right_y) )/len(y)
            
            return p_total, p_left, p_right


        def find_best_split(x, y, split_points):
            loss = [gini_impurity(x, y, point)[0] for point in split_points]
            # for point in split_points:
            #     self.gini_impurity(X, Y, split_point)
            idx = np.argmin(loss)
            return split_points[idx], loss[idx]

        

        loss = 1
        best_split = 0
        best_feature = 0
        for feature in range(X.shape[1]):
            x = X[:, feature]
            split_points = find_split_points(x)
            new_split, new_loss = find_best_split(x, y, split_points)
            if new_loss < loss:
                best_split = new_split
                best_feature = feature
                loss = new_loss
            
        _, loss_left, loss_right = gini_impurity(X[:, best_feature], y, best_split)

        return best_split, best_feature, loss_left, loss_right

    def tree_builder(self, X, y, depth):
        split, feature, loss_left, loss_right = self.splitter(X, y)
        tree = {
            'splitpoint': split,
            'feature': feature,
        }

        left_selector = X[:, feature] < split
        if loss_left == 0 or self.max_depth == depth:
            tree['left'] = np.bincount(y[left_selector]).argmax()
        else:            
            tree['left'] = self.tree_builder(X[left_selector], y[left_selector], depth+1)

        right_selector = X[:, feature] > split
        if loss_right == 0 or self.max_depth == depth:
            tree['right'] = np.bincount(y[right_selector]).argmax()
        else:
            tree['right'] = self.tree_builder(X[right_selector], y[right_selector], depth+1)
        
        return tree

    def fit(self, X, y):
        self.decision_tree = self.tree_builder(X, y, 1)

    #train over data to minimize gini_impurity


# return {
#     splitpoint: best_split
#     feature: feature
#     left: if leaf: _class
#             else: fit()
#     right: {
#         splitpoint:
#         feature:
#         left: _class
#         right: _class
# }
# }


if __name__ == '__main__':
    # import sklearn.datasets

    X = np.array([[1, 2, 5, 6, 7], [6, 8, 5, 4, 2], [9, 2, 3, 4, 5]]).T
    y = np.array([0, 0, 1, 0, 1])

    H = RandomTree(10)
    H.fit(X, y)
    y_hat = H(X)

    print(np.array_equal(y, y_hat))
    print(H.decision_tree)






