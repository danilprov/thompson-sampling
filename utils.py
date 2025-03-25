from sklearn.metrics import mean_squared_error
import numpy as np
import torch


def softmax_torch(x):
    """
    A numerically stable version of the softmax function for 2D tensors.
    """
    exps = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    return exps / torch.sum(exps, dim=1, keepdim=True)

def sigmoid(z):
    # sigmoid function
    return 1 / (1 + np.exp(-z))
