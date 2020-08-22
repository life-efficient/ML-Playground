import numpy as np

def batch_together(examples):
    '''
    Function that batch examples together using vstack
    input: tuple or list of examples
    output: list of vertically stacked examples
    ''' 
    return np.vstack(examples)


examples = ([[1,2,3],[4,5,6],[7,8,9]],
            [[11,12,13],[14,15,16],[17,18,19]])
print(batch_together(examples))

examples = [[1],
            [2]]
print(batch_together(examples))