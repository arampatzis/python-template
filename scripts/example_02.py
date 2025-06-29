#!/usr/bin/env python3

"""Hyper-parameter optimization for training a feed-forward network."""

import torch
import wandb
from torch.utils.data import TensorDataset

from template.trainer import FeedForward, Trainer


def train(config: dict | None = None) -> None:
    """
    Train a feed-forward network for a one-dimensional regression problem.

    Parameters
    ----------
    config : dict | None, optional
        The hyperparameters to use for training. Defaults to None.
    """
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = x**2

    with wandb.init(
        project="train-ff",
        config=config,
    ):
        config_wandb = wandb.config

        trainer = Trainer(
            FeedForward(config_wandb.layer_1, config_wandb.layer_2),
            TensorDataset(x, y),
            batch_size=config_wandb.batch_size,
            lr=config_wandb.lr,
        )

        for epoch in range(config_wandb["epochs"]):
            trainer.step()

            wandb.log(
                {
                    "loss": trainer.losses[-1],
                },
            )

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {trainer.losses[-1]}")


def main() -> None:
    """
    Run a hyperparameter sweep for training a feed-forward network.

    This function creates a new sweep on Weights and Biases with the given
    hyperparameters, and then starts the sweep.
    """
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "loss",
            "goal": "minimize",
        },
        "parameters": {
            "batch_size": {
                "values": [4, 16],
            },
            "lr": {
                "values": [0.01, 0.05, 0.1],
            },
            "layer_1": {
                "values": [
                    10,
                    50,
                ],
            },
            "layer_2": {
                "values": [
                    10,
                    50,
                ],
            },
            "epochs": {
                "value": 150,
            },
        },
    }

    # Create a new sweep
    sweep_id = wandb.sweep(sweep_config, project="train-ff")

    # Start the sweep
    wandb.agent(sweep_id, train, count=100)


if __name__ == "__main__":
    main()
