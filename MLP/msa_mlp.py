from functools import cache, partial

import torch
from data import get_dataloader_copy
from evaluation import accuracy_each_class, objective_fun
from model import MNISTNet, train_mnist_net
from msapy import msa
import torch.optim as optim


def perform_msa_on_model(
    model: MNISTNet,
    num_permutations: int,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    lesion_mode: str = "zero",
):
    if lesion_mode not in {"zero", "mean"}:
        raise ValueError(
            f"Unsupported lesion_mode '{lesion_mode}'. Use 'zero' or 'mean'."
        )

    testloader_copy = get_dataloader_copy(testloader)
    trainloader_copy = get_dataloader_copy(trainloader)

    if lesion_mode == "mean":
        model.cache_layer1_activation_means(trainloader_copy)

    elements = list(range(model.layer1.out_features))

    cached_objective_function = cache(
        partial(
            objective_fun,
            model=model,
            score_fn=accuracy_each_class,
            num_batches=1,
            testloader=testloader_copy,
            lesion_mode=lesion_mode,
        )
    )

    shapley_analysis = msa.interface(
        n_permutations=num_permutations,
        elements=elements,
        objective_function=cached_objective_function,
    )

    return shapley_analysis, model


def perform_one_msa_trial(
    num_permutations:int,
    l1_regularization: bool,
    l2_regularization: bool,
    num_neurons: int,
    num_epochs: int,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    lesion_mode: str = "zero",
):
    if lesion_mode not in {"zero", "mean"}:
        raise ValueError(
            f"Unsupported lesion_mode '{lesion_mode}'. Use 'zero' or 'mean'."
        )

    model = MNISTNet(num_neurons=num_neurons).to(device)
    trainloader_copy = get_dataloader_copy(trainloader)


    if l2_regularization:
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_mnist_net(model, num_epochs, optimizer, trainloader_copy, l1_regularization)
    return perform_msa_on_model(
        model=model,
        num_permutations=num_permutations,
        trainloader=trainloader,
        testloader=testloader,
        lesion_mode=lesion_mode,
    )
