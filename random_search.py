# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def randomSearch(X, labels, function, cost_func, num_epochs, _range, n_features):
    best_weights = None 
    best_bias = None 
    lowest_cost = float('inf') # initialize it very high
    # try num_epochs many different randomised parameterisations
    for i in range(num_epochs):
        w = np.random.randn(-_range, _range, n_features) 
        b = np.random.randn(-_range, _range) 
        # print(w, b)
        y_hat = function(X, w, b) # make prediction
        cost = cost_func(y_hat, labels) # calculate loss
        # if it is the best parameterisation so far then store the weight and bias
        if cost < lowest_cost:
            lowest_cost = cost 
            best_weights = w 
            best_bias = b
    print('Lowest cost of', lowest_cost, 'achieved with weight of', best_weights, 'and bias of', best_bias)
    return best_weights, best_bias 


