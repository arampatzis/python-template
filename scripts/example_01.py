#!/usr/bin/env python3

"""
Train a feed-forward network to learn a simple function.

The function is x^2.

The network is defined as a two-layer feed-forward network with a single
hidden layer. The hidden layer has 5 neurons and uses the ReLU activation
function. The output layer has 1 neuron and uses the identity activation
function.

The network is trained using the mean squared error loss function and the
stochastic gradient descent optimizer.

The training data is a tensor dataset of 100 points evenly spaced between -1
and 1.

The training loop runs for 100 epochs.

The loss is logged to Weights and Biases every epoch.

The script is configured using a configuration dictionary that is passed to
the Trainer constructor. The configuration dictionary has the following keys:

    * lr: The learning rate.
    * batch_size: The batch size.
    * layer_1: The number of neurons in the first layer.
    * layer_2: The number of neurons in the second layer.

The configuration dictionary is logged to Weights and Biases when the script
is initialized.

The script prints the epoch and loss every 10 epochs.
"""

import torch
from torch.utils.data import TensorDataset

import wandb
from template.trainer import FeedForward, Trainer


def main():
    """
    Train a feed-forward network to learn a simple function.

    The function is x^2.

    The network is defined as a two-layer feed-forward network with a single
    hidden layer. The hidden layer has 5 neurons and uses the ReLU activation
    function. The output layer has 1 neuron and uses the identity activation
    function.

    The network is trained using the mean squared error loss function and the
    stochastic gradient descent optimizer.

    The training data is a tensor dataset of 100 points evenly spaced between -1
    and 1.

    The training loop runs for 100 epochs.

    The loss is logged to Weights and Biases every epoch.

    The script prints the epoch and loss every 10 epochs.
    """
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = x**2

    config = {
        "lr": 0.01,
        "batch_size": 4,
        "layer_1": 5,
        "layer_2": 5,
    }

    wandb.init(
        project="train-ff",
        config=config,
    )

    trainer = Trainer(
        FeedForward(config["layer_1"], config["layer_2"]),
        TensorDataset(x, y),
        batch_size=config["batch_size"],
        lr=config["lr"],
    )

    for epoch in range(100):
        trainer.step()

        wandb.log(
            {
                "loss": trainer.losses[-1],
            },
        )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {trainer.losses[-1]}")


if __name__ == "__main__":
    main()
