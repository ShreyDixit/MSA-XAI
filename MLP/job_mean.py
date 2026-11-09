from pathlib import Path

import torch
from data import get_data_loaders


def create_folders_if_not_exist(names: list[Path]):
    for name in names:
        name.mkdir(parents=True, exist_ok=True)


def load_saved_model(model_path: Path, device: torch.device):
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)

    return model.to(device)


def run_mean_lesion_trial(
    model_path: Path,
    output_path: Path,
    num_permutations: int,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    from msa_mlp import perform_msa_on_model

    model = load_saved_model(model_path, device)
    shapley_modes, _ = perform_msa_on_model(
        model=model,
        num_permutations=num_permutations,
        trainloader=trainloader,
        testloader=testloader,
        lesion_mode="mean",
    )
    shapley_modes.save_as_csv(str(output_path))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    shapley_modes_dir = script_dir / "shapley_modes_mean"

    create_folders_if_not_exist([shapley_modes_dir])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_permutations = 1000
    num_neurons_list = [100, 200]

    regularisations = ["l1 regularization", "l2 regularization", "none regularization"]

    trainloader, testloader = get_data_loaders(device)

    for regularisation in regularisations:
        regularisation_models_dir = models_dir / regularisation
        regularisation_output_dir = shapley_modes_dir / regularisation
        create_folders_if_not_exist([regularisation_output_dir])

        for num_neurons in num_neurons_list:
            source_dir = regularisation_models_dir / str(num_neurons)
            output_dir = regularisation_output_dir / str(num_neurons)
            create_folders_if_not_exist([output_dir])

            model_paths = sorted(source_dir.glob("trial_*.pt"))
            if not model_paths:
                print(f"Skipping {source_dir}: no saved models found.")
                continue

            for model_path in model_paths:
                output_path = output_dir / f"{model_path.stem}.csv"
                if output_path.exists():
                    print(f"Skipping {output_path}: mean-lesion result already exists.")
                    continue

                print(f"Processing {model_path} -> {output_path}")
                run_mean_lesion_trial(
                    model_path=model_path,
                    output_path=output_path,
                    num_permutations=num_permutations,
                    trainloader=trainloader,
                    testloader=testloader,
                    device=device,
                )


if __name__ == "__main__":
    main()
