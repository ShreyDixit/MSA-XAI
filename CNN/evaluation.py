from typing import Optional

import torch


@torch.inference_mode()
def evaluate_model(
    model,
    testloader,
    lesion_idx: Optional[tuple[tuple[int, int], ...]] = None,
    num_batches: int = -1,
    score_fn=None,
    device: torch.device = torch.device("cpu"),
):
    """Return a score computed on the test dataset."""

    if score_fn is None:
        score_fn = accuracy_score

    model.eval()

    targets = []
    preds = []

    for i, data in enumerate(testloader):
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images, lesion_idx)
        predicted = outputs.argmax(dim=1)

        preds.append(predicted)
        targets.append(labels)

        if i == (num_batches - 1):
            break

    return score_fn(torch.cat(targets), torch.cat(preds))


def accuracy_score(targets: torch.Tensor, preds: torch.Tensor) -> float:
    return (targets == preds).float().mean().item()


def accuracy_each_class(
    targets: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int = 10,
):
    targets = targets.to(torch.int64)
    preds = preds.to(torch.int64)

    flat_indices = targets * num_classes + preds
    matrix = torch.bincount(
        flat_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    matrix = matrix.to(torch.float32)

    per_class_accuracy = matrix.diag() / (matrix.sum(dim=1) + 1e-6)
    return per_class_accuracy.cpu().numpy()


def preload_batches(
    dataloader,
    num_batches: int = 1,
    device: torch.device = torch.device("cpu"),
):
    images = []
    labels = []

    for i, data in enumerate(dataloader):
        batch_images, batch_labels = data
        images.append(batch_images.to(device, non_blocking=True))
        labels.append(batch_labels.to(device, non_blocking=True))

        if i == (num_batches - 1):
            break

    return torch.cat(images), torch.cat(labels)


@torch.inference_mode()
def evaluate_preloaded_batch(
    model,
    batch_data,
    lesion_idx: Optional[tuple[tuple[int, int], ...]] = None,
    score_fn=None,
):
    if score_fn is None:
        score_fn = accuracy_score

    model.eval()

    images, labels = batch_data
    outputs = model(images, lesion_idx)
    predicted = outputs.argmax(dim=1)
    return score_fn(labels, predicted)


def objective_fun(
    lesion_idx,
    model,
    testloader=None,
    num_batches: int = -1,
    score_fn=None,
    device: torch.device = torch.device("cpu"),
    batch_data=None,
):
    if batch_data is not None:
        return evaluate_preloaded_batch(model, batch_data, lesion_idx, score_fn)

    return evaluate_model(model, testloader, lesion_idx, num_batches, score_fn, device)
