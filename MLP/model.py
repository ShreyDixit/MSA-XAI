from typing import Optional, Tuple, Union
import torch.nn as nn
import torch


class MNISTNet(nn.Module):
    def __init__(self, num_neurons: int = 10):
        super(MNISTNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, num_neurons)
        self.layer2 = nn.Sequential(nn.LeakyReLU(), nn.Linear(num_neurons, 10))

    def forward(
        self, x: torch.Tensor, lesion_idx: Optional[Union[int, Tuple[int]]] = None
    ) -> torch.Tensor:
        """forward function to calculate the scores for each class

        Args:
            x (torch.Tensor): data of shape [batch_size, 28*28]
            lesion_idx (Optional[Union[int,List[int]]], optional): the neuron that we want to lesion in the hidden layer1. Defaults to None i.e. no lesioning performed.

        Returns:
            torch.Tensor: scores for each class
        """
        out = self.layer1(x)

        if lesion_idx:
            out[:, lesion_idx] = 0  # set the value to 0 for the lesioned neuron

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
