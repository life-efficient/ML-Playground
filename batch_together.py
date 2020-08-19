def batch_data(*args, batch_size):
    """ Function to group data into batches
     Use:
        X_batche = batch_data(X, batch_size = 8)
        X_batche, y_batche = batch_data(X, y, batch_size = 8)
     
     Args:
        args*: Single dataset or tuple of datasets 
        (size of a dataset = num_examples_in_dataset x num_features_per_example)
        batch_size ([int]): number examples in batch

    Returns:
        List of batches for single dataset or tuple of list of batches for several datasets
        (size of list of batches = num_batches_that_result x num_examples_per_batch x num_features_per_example)
    """

    # Check that batch size is more than 0
    assert batch_size > 0, 'batch_size should be integer > 0'
    
    # Check that the inputs are of the same length
    for arg in args:
        assert len(arg) == len(args[0]), 'different lengths of input arguments'


    res = []    
    for arg in args:
        idx = 0
        batches = []
        m = len(args[0]) 
        while idx < m:
            if idx + batch_size <= m:  # if this batch finishes before we reach the end of the dataset
                batch = arg[idx:idx + batch_size]
            else:  # if this batch tries to index beyond the length of the dataset
                batch = arg[idx:]
            batches.append(batch)
            idx += batch_size
        res.append(batches)
    
    if len(args) > 1:
        res = tuple(res)
    else:
        res = res[0]

    return res

# #Tests:
# X = [1,2,3,4,5]
# X_b = batch_data(X, batch_size=3)
# print('X_b = ', X_b)
# print()

# X = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
# y = [0, 1, 0, 1, 1] 
# X_b, y_b = batch_data(X, y, batch_size=2)
# print('X_b = ', X_b)
# print('y_b = ', y_b)
# print()

# import numpy as np
# X = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]])
# y = np.array([0,1,0,1,1])
# print(X.shape)
# print(y.shape)

# X_b, y_b = batch_data(X, y, batch_size=2)
# print('X_b = ', X_b)
# print('y_b = ', y_b)