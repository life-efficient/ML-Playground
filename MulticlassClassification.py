import torch
from data import get_cla

class MC(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_classes)


    def __call__(self, x):
        x = self.linear(x)
        x = torch.nn.functional.softmax(x)
        return x

X, Y = 
mc = MC()

epochs = 10
for e in range(epochs):
