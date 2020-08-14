import numpy as np

class LinearRegression:
    def __init__(self): # initalize parameters 
        self.w = np.random.randn() ## randomly initialise weight
        self.b = np.random.randn() ## randomly initialise bias
        
    def __call__(self, X): # how do we calculate output from an input in our model?
        ypred = self.w * X + self.b ## make a prediction using a linear hypothesis
        return ypred # return prediction
    
    def update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's weights to the new weight value passed to the function
        self.b = new_b ## do the same for the bias

    def loss(self, Y, y_hat= None, X= None):
        if y_hat is None:
            y_hat = self(X)
        diff = y_hat - Y
        squares = np.square(diff)
        mean = np.mean(squares)
        return mean

    def grid_search(self, X, Y, y_hat=None, limit= 20, step=1):
        if y_hat is None:
            y_hat = self(X)
        best_weights = 0 ## no best weight found yet
        best_bias = 0 ## no best bias found yet
        lowest_cost = float('inf') ## initialize it very high (how high can it be?)
        for w in range(-limit, limit, step):
            for b in range(-limit, limit, step): ## try this many different parameterisations
                self.update_params(w, b) ## update our model with these random parameters
                y_hat = self(X) ## make prediction
                cost = self.loss(y_hat, Y) ## calculate loss
                if cost < lowest_cost: ## if this is the best parameterisation so far
                    lowest_cost = cost ## update the lowest running cost to the cost for this parameterisation
                    best_weights = w ## get best weights so far from the model
                    best_bias = b ## get best bias so far from the model
        print('Lowest cost of', lowest_cost, 'achieved with weight of', best_weights, 'and bias of', best_bias)
        return best_weights, best_bias ## return the best weight and best bias