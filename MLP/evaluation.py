from typing import Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch

@torch.no_grad()
def evaluate_model(model, testloader, lesion_idx: Optional[Union[int, Tuple[int]]] = None, num_batches: int = -1, score_fn=accuracy_score):
    """return the accuracy of the model on test dataset

    Args:
        lesion_idx (Optional[Union[int,List[int]]], optional): the neuron that we want to lesion in the hidden layer1. Defaults to None i.e. no lesioning performed.
        num_batches (int, optional): the number of batches we want to test our model. Defaults to -1 i.e. all data

    Returns:
        float: test accuracy
    """

    targets = []
    preds = []
    for i, data in enumerate(testloader):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images, lesion_idx)
        # the class with the highest score is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        preds.append(predicted)
        targets.append(labels)

        if i == (num_batches-1):
            break

    return score_fn(torch.concat(preds).cpu(), torch.concat(targets).cpu())

def accuracy_each_class(targets, preds):
    matrix = confusion_matrix(targets, preds)
    acc = []
    for i, val in enumerate(matrix.diagonal()/(matrix.sum(axis=1)+ 1e-6)):
        acc.append(val)

    return np.array(acc)

def objective_fun(lesion_idx, model, testloader, num_batches, score_fn):
    return evaluate_model(model, testloader, lesion_idx, num_batches, score_fn)