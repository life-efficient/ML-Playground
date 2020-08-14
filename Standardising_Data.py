

def standardize_data(dataset):

    mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0) ## get mean and standard deviation of dataset
    standardized_dataset  = (dataset-mean)/std
    print("Here is the shape of your standardised Data : ", stanadrdized_dataset.shape)

    return standardized_dataset

