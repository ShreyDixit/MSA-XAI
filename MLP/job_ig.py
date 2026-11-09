import csv
from pathlib import Path

import torch
from captum.attr import LayerIntegratedGradients
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


def save_matrix_as_csv(output_path: Path, matrix: torch.Tensor) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as output_file:
        output_file.write(f"#shape={matrix.shape[0]}\n")
        writer = csv.writer(output_file)
        writer.writerow([""] + list(range(matrix.shape[1])))
        for class_idx, row in enumerate(matrix.cpu().tolist()):
            writer.writerow([class_idx] + row)


def compute_class_specific_ig(
    model,
    testloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    n_steps: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    lig = LayerIntegratedGradients(model, model.layer1)

    signed_sums = torch.zeros(
        num_classes, model.layer1.out_features, device=device, dtype=torch.float32
    )
    absolute_sums = torch.zeros_like(signed_sums)
    class_counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    for images, labels in testloader:
        images = images.detach().requires_grad_(True)
        baselines = torch.zeros_like(images)
        attributions = lig.attribute(
            images,
            baselines=baselines,
            target=labels,
            n_steps=n_steps,
            internal_batch_size=images.shape[0],
            attribute_to_layer_input=False,
        )

        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if not class_mask.any():
                continue

            class_attributions = attributions[class_mask]
            signed_sums[class_idx] += class_attributions.sum(dim=0)
            absolute_sums[class_idx] += class_attributions.abs().sum(dim=0)
            class_counts[class_idx] += class_mask.sum()

    class_counts = class_counts.clamp_min(1).unsqueeze(1)

    mean_signed = signed_sums / class_counts
    mean_absolute = absolute_sums / class_counts

    return mean_signed.cpu(), mean_absolute.cpu()


def run_ig_trial(
    model_path: Path,
    signed_output_path: Path,
    absolute_output_path: Path,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_steps: int = 50,
) -> None:
    model = load_saved_model(model_path, device)
    mean_signed, mean_absolute = compute_class_specific_ig(
        model=model,
        testloader=testloader,
        n_steps=n_steps,
    )
    if not signed_output_path.exists():
        save_matrix_as_csv(signed_output_path, mean_signed)

    if not absolute_output_path.exists():
        save_matrix_as_csv(absolute_output_path, mean_absolute)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    ig_signed_dir = script_dir / "ig_class_specific_signed"
    ig_absolute_dir = script_dir / "ig_class_specific_absolute"

    create_folders_if_not_exist([ig_signed_dir, ig_absolute_dir])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_neurons_list = [200]
    n_steps = 200

    regularisations = ["l1 regularization", "l2 regularization", "none regularization"]

    _, testloader = get_data_loaders(device)

    for regularisation in regularisations:
        regularisation_models_dir = models_dir / regularisation
        regularisation_signed_dir = ig_signed_dir / regularisation
        regularisation_absolute_dir = ig_absolute_dir / regularisation
        create_folders_if_not_exist(
            [regularisation_signed_dir, regularisation_absolute_dir]
        )

        for num_neurons in num_neurons_list:
            source_dir = regularisation_models_dir / str(num_neurons)
            signed_output_dir = regularisation_signed_dir / str(num_neurons)
            absolute_output_dir = regularisation_absolute_dir / str(num_neurons)
            create_folders_if_not_exist([signed_output_dir, absolute_output_dir])

            model_paths = sorted(source_dir.glob("trial_*.pt"))
            if not model_paths:
                print(f"Skipping {source_dir}: no saved models found.")
                continue

            for model_path in model_paths:
                signed_output_path = signed_output_dir / f"{model_path.stem}.csv"
                absolute_output_path = absolute_output_dir / f"{model_path.stem}.csv"

                if signed_output_path.exists() and absolute_output_path.exists():
                    print(
                        "Skipping "
                        f"{model_path}: class-specific IG results already exist."
                    )
                    continue

                print(
                    f"Processing {model_path} -> "
                    f"{signed_output_path} and {absolute_output_path}"
                )
                run_ig_trial(
                    model_path=model_path,
                    signed_output_path=signed_output_path,
                    absolute_output_path=absolute_output_path,
                    testloader=testloader,
                    device=device,
                    n_steps=n_steps,
                )


if __name__ == "__main__":
    main()
