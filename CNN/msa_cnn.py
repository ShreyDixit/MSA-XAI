from functools import cache, partial

import torch
import torch.optim as optim
from msapy import msa

from evaluation import (
    accuracy_each_class,
    objective_fun,
    preload_batches,
    preload_hidden_batch,
)
from model import LesionableResNet18, train_imagenette_net


REGULARIZATION_LAMBDA = 3e-4
LEARNING_RATE = 0.01


def perform_one_msa_trial(
    num_permutations: int,
    l1_regularization: bool,
    l2_regularization: bool,
    num_epochs: int,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model = LesionableResNet18(
        hidden_features=200,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    weight_decay = REGULARIZATION_LAMBDA if l2_regularization else 0.0
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=LEARNING_RATE * 0.01,
    )

    train_imagenette_net(
        model,
        num_epochs,
        optimizer,
        trainloader,
        device,
        valloader=testloader,
        l1_regularization=l1_regularization,
        l1_lambda=REGULARIZATION_LAMBDA,
        scheduler=scheduler,
    )

    elements = model.lesionable_elements()
    eval_batch = preload_batches(testloader, num_batches=1, device=device)
    eval_hidden_batch = preload_hidden_batch(model, eval_batch)

    cached_objective_function = cache(
        partial(
            objective_fun,
            model=model,
            score_fn=accuracy_each_class,
            hidden_batch=eval_hidden_batch,
        )
    )

    shapley_analysis = msa.interface(
        n_permutations=num_permutations,
        elements=elements,
        objective_function=cached_objective_function,
        dual_progress_bars=False
    )

    return shapley_analysis, model
