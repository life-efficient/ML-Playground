#%%
import numpy as np
from time import sleep


#%%
def compute_distances(input_X, dataset_X):
    """Takes in an array of inputs and finds each of their distances from every example in a dataset"""
    distances = np.zeros((len(input_X), len(dataset_X))) ## matrix of distances between each input x and each x in our dataset
    for i, this_x in enumerate(input_X): ## enumerate over input
        for j, another_x in enumerate(dataset_X): ## enumerate over dataset
            distance = np.linalg.norm(this_x - another_x) ## compute euclidian distance
            distances[i][j] = distance
    return distances

# %%
class DBScan:# defining class DBSCAN

    def __init__(self, X , epsilon= 3, d= 1.0):# initialize with epsilon and d
        self.epsilon = epsilon
        self.d = d
        self.X = X
        
        self.close_enough = None
        self.category_list = None
        self.cluster_list = np.zeros(len(self.X))

    def cluster_search(self, idx, cluster):
        if (self.cluster_list[idx] == 0) and (self.category_list[idx] != 2):
            self.cluster_list[idx] = cluster
            neighbor_list = np.where(self.close_enough[idx])[0]
            for neighbor_idx in neighbor_list:
                self.cluster_search(neighbor_idx, cluster)

    
    def __call__(self):
        distances = compute_distances(self.X, self.X)
        self.close_enough = distances <= self.d
        how_many_neighbors = self.close_enough.sum(axis=1)         # matrix of distances between datapoints
        self.category_list = np.zeros_like(how_many_neighbors)     #noncore
        self.category_list[how_many_neighbors > self.epsilon] = 1  #core
        self.category_list[how_many_neighbors == 0] = 2            #noise
        
        if (self.category_list == 0).any():
            for idx in np.where(self.category_list == 0)[0]:                           #iterates over noncore points
                if not (self.category_list[self.close_enough[idx]] == 1).any():  #checks if any neighbors are core points
                    self.category_list[idx] = 2                                     #if not, noncore point is converted in noise point

        # print(self.category_list)
        cluster = 1
        for idx in range(len(self.X)):
            # print(self.category_list[idx])
            # print(self.cluster_list[idx])
            if (self.category_list[idx] == 1) and (self.cluster_list[idx] == 0):
                self.cluster_search(idx, cluster)

                cluster += 1
        
        return self.cluster_list.astype(int)

            


if __name__ == '__main__':

#%%
    from vis import show_data
    from sklearn import datasets

    X, Y = datasets.make_moons( )

    show_data(X, Y)

#%%
    H = DBScan(X=X, d=0.5)
    predictions = H()
    show_data(X, predictions)
    

# %%
