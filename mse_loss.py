import numpy as np

def mse_loss(y_predict, y_true):
    '''
    Function to calculate mean squared error between input and target
    input: y_predict, y_true
    output: loss_value
    '''
    assert len(y_predict) == len(y_true), 'inputs lengths to mse_loss are different' 

    y_predict = np.array(y_predict)
    y_true = np.array(y_true)

    return np.mean((y_predict - y_true)**2)


'''
# Test
y_predict = [1,2,3]
y_true = [0,0,0]
print(mse_loss(y_predict, y_true))
'''