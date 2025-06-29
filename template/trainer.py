"""
Contains the Trainer class for training a model on a dataset.

The Trainer class uses Weights and Biases to log the loss at each epoch.
"""

import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader


class FeedForward(nn.Module):
    """
    A feed-forward neural network with two layers.

    Parameters
    ----------
    layer_1 : int
        The number of neurons in the first layer.
    layer_2 : int
        The number of neurons in the second layer.

    Attributes
    ----------
    fc : nn.Sequential
        The neural network.

    Examples
    --------
    >>> model = FeedForward(10, 10)
    >>> model(torch.randn(10))
    """

    def __init__(self, layer_1: int, layer_2: int) -> None:
        """
        Initialize a feed-forward neural network with two layers.

        Parameters
        ----------
        layer_1 : int
            The number of neurons in the first layer.
        layer_2 : int
            The number of neurons in the second layer.
        """
        super().__init__()
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(1, layer_1),
            nn.Tanh(),
            nn.Linear(layer_1, layer_2),
            nn.Tanh(),
            nn.Linear(layer_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.fc(x)  # type: ignore[no-any-return]


class Trainer:
    """
    Train a model on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    dataset : torch.utils.data.Dataset
        The dataset to train on.
    batch_size : int, optional
        The batch size. Defaults to 16.
    lr : float, optional
        The learning rate. Defaults to 0.01.

    Attributes
    ----------
    model : torch.nn.Module
        The model being trained.
    loader : torch.utils.data.DataLoader
        The data loader for the dataset.
    optimizer : torch.optim.Optimizer
        The optimizer being used to train the model.
    losses : list
        The average loss at each epoch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 16,
        lr: float = 0.01,
    ) -> None:
        """
        Initialize the trainer.

        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        dataset : torch.utils.data.Dataset
            The dataset to train on.
        batch_size : int, optional
            The batch size. Defaults to 16.
        lr : float, optional
            The learning rate. Defaults to 0.01.
        """
        self.model = model
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.SGD(model.parameters(), lr=lr)

        self.losses: list[float] = []

        wandb.watch(self.model)

    def step(self) -> None:
        """
        Train the model for a single epoch.

        Compute the loss for each batch in the dataset, backpropagate,
        and update the model parameters using the optimizer. The average
        loss over all batches is stored in the `losses` attribute.
        """
        epoch_loss: float = 0
        for batch_x, batch_y in self.loader:
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = nn.MSELoss()(output, batch_y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(self.loader)
        self.losses.append(epoch_loss)
