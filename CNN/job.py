import os

import torch
from joblib import Parallel, delayed

from data import get_data_loaders
from msa_cnn import perform_one_msa_trial


def create_folders_if_not_exist(names: list[str]):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)


models_dir = "CNN/models"
shapley_modes_dir = "CNN/shapley_modes"

create_folders_if_not_exist([models_dir, shapley_modes_dir])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_permutations = 1000
num_epochs = 50
num_trials = 10
n_jobs = 10

regularisations = ["l1 regularization", "l2 regularization", "none regularization"]

trainloader, testloader = get_data_loaders(train_num_workers=0, test_num_workers=0)

if device.type == "cuda":
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
else:
    print(f"Using device: {device}")

print(
    "Train DataLoader: "
    f"batch_size={trainloader.batch_size}, workers={trainloader.num_workers}, "
    f"pin_memory={trainloader.pin_memory}"
)
print(
    "Test DataLoader: "
    f"batch_size={testloader.batch_size}, workers={testloader.num_workers}, "
    f"pin_memory={testloader.pin_memory}"
)

for regularisation in regularisations:
    create_folders_if_not_exist(
        [f"{models_dir}/{regularisation}", f"{shapley_modes_dir}/{regularisation}"]
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(perform_one_msa_trial)(
            num_permutations,
            regularisation == "l1 regularization",
            regularisation == "l2 regularization",
            num_epochs,
            trainloader,
            testloader,
            device,
        )
        for _ in range(num_trials)
    )

    for i, result in enumerate(results):
        shapley_modes, model = result
        shapley_modes.save_as_csv(f"{shapley_modes_dir}/{regularisation}/trial_{i}.csv")
        torch.save(model, f"{models_dir}/{regularisation}/trial_{i}.pt")
