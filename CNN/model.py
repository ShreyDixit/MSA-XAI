from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from evaluation import evaluate_model


LesionIndex = Optional[Union[int, tuple[int, ...]]]


def _normalize_lesion_idx(lesion_idx: LesionIndex) -> tuple[int, ...]:
    if lesion_idx is None:
        return ()

    if isinstance(lesion_idx, int):
        return (lesion_idx,)

    return tuple(lesion_idx)


class LesionableResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        hidden_features: int = 200,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except TypeError:
            backbone = resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.last_linear = nn.Linear(in_features, hidden_features)
        self.output_layer = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_features, num_classes),
        )
        self.hidden_features = hidden_features
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

    def lesionable_elements(self) -> list[int]:
        return list(range(self.hidden_features))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def compute_hidden(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.last_linear(features)

    def forward_from_hidden(
        self,
        hidden: torch.Tensor,
        lesion_idx: LesionIndex = None,
    ) -> torch.Tensor:
        out = hidden

        lesion_indices = _normalize_lesion_idx(lesion_idx)
        if lesion_indices:
            out = out.clone()
            out[:, lesion_indices] = 0

        return self.output_layer(out)

    def forward(
        self,
        x: torch.Tensor,
        lesion_idx: LesionIndex = None,
    ) -> torch.Tensor:
        hidden = self.compute_hidden(x)
        return self.forward_from_hidden(hidden, lesion_idx)


TinyResNet = LesionableResNet18


def train_imagenette_net(
    model: nn.Module,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    valloader: Optional[torch.utils.data.DataLoader] = None,
    l1_regularization: bool = False,
    l1_lambda: float = 1e-4,
    scheduler=None,
):
    loss_func = nn.CrossEntropyLoss()
    best_epoch = 0
    best_val_accuracy = float("-inf")
    best_state_dict = None

    for epoch_idx in range(epochs):
        model.train()
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs, labels)

            if l1_regularization:
                l1_norm = sum(
                    parameter.abs().sum()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

        if valloader is not None:
            val_accuracy = evaluate_model(model, valloader, device=device)
            if val_accuracy > best_val_accuracy:
                best_epoch = epoch_idx + 1
                best_val_accuracy = val_accuracy
                best_state_dict = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            print(f"Epoch {epoch_idx + 1}/{epochs} - val accuracy: {val_accuracy:.4f}")

        if scheduler is not None:
            scheduler.step()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"Restored best model from epoch {best_epoch}/{epochs} "
            f"- val accuracy: {best_val_accuracy:.4f}"
        )
