from typing import Optional, Union
import torch.nn as nn
import torch


class MNISTNet(nn.Module):
    def __init__(self, num_neurons: int = 10):
        super(MNISTNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, num_neurons)
        self.layer2 = nn.Sequential(nn.LeakyReLU(), nn.Linear(num_neurons, 10))
        self.register_buffer(
            "layer1_activation_means",
            torch.full((num_neurons,), float("nan")),
        )

    def _ensure_activation_mean_buffer(self) -> None:
        if hasattr(self, "layer1_activation_means"):
            return

        self.register_buffer(
            "layer1_activation_means",
            torch.full(
                (self.layer1.out_features,),
                float("nan"),
                device=self.layer1.weight.device,
            ),
        )

    @staticmethod
    def _has_lesion_indices(
        lesion_idx: Optional[Union[int, tuple[int, ...], list[int]]]
    ) -> bool:
        if lesion_idx is None:
            return False

        if isinstance(lesion_idx, (tuple, list)):
            return len(lesion_idx) > 0

        return True

    @staticmethod
    def _normalize_lesion_indices(
        lesion_idx: Union[int, tuple[int, ...], list[int]]
    ) -> Union[int, list[int]]:
        if isinstance(lesion_idx, tuple):
            return list(lesion_idx)

        return lesion_idx

    @torch.no_grad()
    def cache_layer1_activation_means(
        self, dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Cache the mean pre-activation of each hidden neuron over a dataloader."""
        self._ensure_activation_mean_buffer()
        was_training = self.training
        device = self.layer1.weight.device
        activation_sum = torch.zeros_like(self.layer1_activation_means)
        num_examples = 0

        self.eval()
        for inputs, _ in dataloader:
            hidden = self.layer1(inputs.to(device))
            activation_sum += hidden.sum(dim=0)
            num_examples += hidden.shape[0]

        if num_examples == 0:
            raise ValueError("Cannot cache activation means from an empty dataloader.")

        self.layer1_activation_means.copy_(activation_sum / num_examples)

        if was_training:
            self.train()

    def forward(
        self,
        x: torch.Tensor,
        lesion_idx: Optional[Union[int, tuple[int, ...], list[int]]] = None,
        lesion_mode: str = "zero",
    ) -> torch.Tensor:
        """forward function to calculate the scores for each class

        Args:
            x (torch.Tensor): data of shape [batch_size, 28*28]
            lesion_idx (Optional[Union[int,List[int]]], optional): the neuron that we want to lesion in the hidden layer1. Defaults to None i.e. no lesioning performed.
            lesion_mode (str, optional): how to lesion hidden neurons. Use "zero"
                to clamp the selected neurons to 0 or "mean" to replace them with
                their cached mean pre-activation. Defaults to "zero".

        Returns:
            torch.Tensor: scores for each class
        """
        out = self.layer1(x)

        if self._has_lesion_indices(lesion_idx):
            lesion_indices = self._normalize_lesion_indices(lesion_idx)
            if lesion_mode == "zero":
                out[:, lesion_indices] = 0
            elif lesion_mode == "mean":
                self._ensure_activation_mean_buffer()
                lesion_mean = self.layer1_activation_means[lesion_indices]
                if torch.isnan(lesion_mean).any():
                    raise ValueError(
                        "Mean replacement requested before activation means were cached."
                    )
                out[:, lesion_indices] = lesion_mean
            else:
                raise ValueError(
                    f"Unsupported lesion_mode '{lesion_mode}'. Use 'zero' or 'mean'."
                )

        return self.layer2(out)


def train_mnist_net(model, epochs, optimizer, trainloader, l1_regularization=False):
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            if l1_regularization:
                l1_lambda = 0.0001
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
