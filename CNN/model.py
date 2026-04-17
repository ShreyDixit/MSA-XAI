from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn

from evaluation import evaluate_model


def lesion_list_to_lesion_dict(
    lesion_idx: Optional[tuple[tuple[int, int], ...]]
) -> defaultdict[int, list[int]]:
    lesion_dict = defaultdict(list)
    if lesion_idx is None:
        return lesion_dict

    for block_idx, channel_idx in lesion_idx:
        lesion_dict[block_idx].append(channel_idx)

    return lesion_dict


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual

        return self.activation(out)


class TinyResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        block_channels: tuple[int, ...] = (16, 32, 48, 64, 88,),
    ):
        super().__init__()

        stem_channels = block_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        in_channels = stem_channels
        blocks = []
        for i, out_channels in enumerate(block_channels):
            stride = 1 if i == 0 else 2
            blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels

        self.residual_blocks = nn.ModuleList(blocks)
        self.block_channels = block_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def lesionable_elements(self) -> list[tuple[int, int]]:
        return [
            (block_idx, channel_idx)
            for block_idx, num_channels in enumerate(self.block_channels)
            for channel_idx in range(num_channels)
        ]

    def forward(
        self,
        x: torch.Tensor,
        lesion_idx: Optional[tuple[tuple[int, int], ...]] = None,
    ) -> torch.Tensor:
        lesion_dict = lesion_list_to_lesion_dict(lesion_idx)

        x = self.stem(x)
        for block_idx, block in enumerate(self.residual_blocks):
            x = block(x)
            if block_idx in lesion_dict:
                x = x.clone()
                x[:, lesion_dict[block_idx]] = 0

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_imagenette_net(
    model: TinyResNet,
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
                l1_norm = sum(parameter.abs().sum() for parameter in model.parameters())
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
