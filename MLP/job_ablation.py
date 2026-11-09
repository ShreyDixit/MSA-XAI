import argparse
import csv
from pathlib import Path

import torch
from data import get_data_loaders
from evaluation import accuracy_each_class, evaluate_model


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


def compute_class_specific_ablation_contributions(
    model,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    lesion_mode: str = "zero",
) -> torch.Tensor:
    if lesion_mode not in {"zero", "mean"}:
        raise ValueError(
            f"Unsupported lesion_mode '{lesion_mode}'. Use 'zero' or 'mean'."
        )

    model.eval()

    if lesion_mode == "mean":
        model.cache_layer1_activation_means(trainloader)

    baseline_scores = torch.as_tensor(
        evaluate_model(
            model,
            testloader,
            score_fn=accuracy_each_class,
            lesion_mode=lesion_mode,
        ),
        dtype=torch.float32,
    )

    num_neurons = model.layer1.out_features
    contributions = torch.empty(
        baseline_scores.shape[0],
        num_neurons,
        dtype=torch.float32,
    )

    for neuron_idx in range(num_neurons):
        ablated_scores = torch.as_tensor(
            evaluate_model(
                model,
                testloader,
                lesion_idx=neuron_idx,
                score_fn=accuracy_each_class,
                lesion_mode=lesion_mode,
            ),
            dtype=torch.float32,
        )
        contributions[:, neuron_idx] = baseline_scores - ablated_scores

    return contributions


def run_ablation_trial(
    model_path: Path,
    signed_output_path: Path,
    absolute_output_path: Path,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    lesion_mode: str = "zero",
) -> None:
    model = load_saved_model(model_path, device)
    contributions = compute_class_specific_ablation_contributions(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        lesion_mode=lesion_mode,
    )

    if not signed_output_path.exists():
        save_matrix_as_csv(signed_output_path, contributions)

    if not absolute_output_path.exists():
        save_matrix_as_csv(absolute_output_path, contributions.abs())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute class-specific single-neuron ablation contributions for saved "
            "MLP MNIST models."
        )
    )
    parser.add_argument(
        "--num-neurons",
        nargs="+",
        type=int,
        default=[50, 100, 200],
        help="Neuron counts to process. Defaults to the same counts as job_ig.py.",
    )
    parser.add_argument(
        "--lesion-mode",
        choices=["zero", "mean"],
        default="zero",
        help="How to ablate hidden neurons. Defaults to zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"

    lesion_suffix = "" if args.lesion_mode == "zero" else "_mean"
    ablation_signed_dir = (
        script_dir / f"single_neuron_ablation{lesion_suffix}_class_specific_signed"
    )
    ablation_absolute_dir = (
        script_dir / f"single_neuron_ablation{lesion_suffix}_class_specific_absolute"
    )

    create_folders_if_not_exist([ablation_signed_dir, ablation_absolute_dir])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regularisations = ["l1 regularization", "l2 regularization", "none regularization"]

    trainloader, testloader = get_data_loaders(device)

    for regularisation in regularisations:
        regularisation_models_dir = models_dir / regularisation
        regularisation_signed_dir = ablation_signed_dir / regularisation
        regularisation_absolute_dir = ablation_absolute_dir / regularisation
        create_folders_if_not_exist(
            [regularisation_signed_dir, regularisation_absolute_dir]
        )

        for num_neurons in args.num_neurons:
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
                        f"{model_path}: single-neuron ablation results already exist."
                    )
                    continue

                print(
                    f"Processing {model_path} -> "
                    f"{signed_output_path} and {absolute_output_path}"
                )
                run_ablation_trial(
                    model_path=model_path,
                    signed_output_path=signed_output_path,
                    absolute_output_path=absolute_output_path,
                    trainloader=trainloader,
                    testloader=testloader,
                    device=device,
                    lesion_mode=args.lesion_mode,
                )


if __name__ == "__main__":
    main()
