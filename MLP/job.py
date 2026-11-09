import os
from data import get_data_loaders
import torch
from joblib import Parallel, delayed
from msa_mlp import perform_one_msa_trial


def create_folders_if_not_exist(names: list[str]):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)


device = torch.device("cuda")
num_permutations = 1000
num_neurons_list = list(range(5, 201, 50))
num_epochs = 6
num_trials = 10
lesion_mode = "zero"

if lesion_mode == "mean":
    models_dir = "models_mean"
    shapley_modes_dir = "shapley_modes_mean"
else:
    models_dir = "models"
    shapley_modes_dir = "shapley_modes"

create_folders_if_not_exist([models_dir, shapley_modes_dir])

regularisations = ["l1 regularization", "l2 regularization", "none regularization"]

trainloader, testloader = get_data_loaders(device)

for regularisation in regularisations:
    create_folders_if_not_exist(
        [f"{models_dir}/{regularisation}", f"{shapley_modes_dir}/{regularisation}"]
    )
    for num_neurons in num_neurons_list:
        create_folders_if_not_exist(
            [
                f"{models_dir}/{regularisation}/{num_neurons}",
                f"{shapley_modes_dir}/{regularisation}/{num_neurons}",
            ]
        )
        results = Parallel(n_jobs=8)(
            delayed(perform_one_msa_trial)(
                num_permutations,
                regularisation == "l1 regularization",
                regularisation == "l2 regularization",
                num_neurons,
                num_epochs,
                trainloader,
                testloader,
                device,
                lesion_mode,
            )
            for _ in range(num_trials)
        )

        for i, result in enumerate(results):
            shapleu_modes, model = result
            shapleu_modes.save_as_csv(
                f"{shapley_modes_dir}/{regularisation}/{num_neurons}/trial_{i}.csv"
            )
            torch.save(
                model, f"{models_dir}/{regularisation}/{num_neurons}/trial_{i}.pt"
            )
