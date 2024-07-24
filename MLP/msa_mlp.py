from dataclasses import dataclass
from functools import cache, partial

import pandas as pd
import torch
from data import get_dataloader_copy
from evaluation import accuracy_each_class, objective_fun
from model import MNISTNet, train_mnist_net
from msapy import msa
import torch.optim as optim


def perform_one_msa_trial(
    num_permutations:int,
    l1_regularization: bool,
    l2_regularization: bool,
    num_neurons: int,
    num_epochs: int,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model = MNISTNet(num_neurons=num_neurons).to(device)
    testloader_copy = get_dataloader_copy(testloader)
    trainloader_copy = get_dataloader_copy(trainloader)


    if l2_regularization:
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_mnist_net(model, num_epochs, optimizer, trainloader_copy, l1_regularization)

    elements = list(range(num_neurons))
        
    cached_objective_function = cache(
        partial(
            objective_fun,
            model=model,
            score_fn=accuracy_each_class,
            num_batches=1,
            testloader=testloader_copy,
        )
    )

    shapley_analysis = msa.interface(
        n_permutations=num_permutations,
        elements=elements,
        objective_function=cached_objective_function,
    )

    return shapley_analysis, model