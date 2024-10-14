#!/usr/bin/env python3

"""Train a feed-forward network to learn a simple function."""

import torch
from torch.utils.data import TensorDataset

import wandb
from template.trainer import FeedForward, Trainer


def train(config=None):
    """
    Train a feed-forward network to learn a simple function.

    Parameters
    ----------
    config : dict, optional
        The hyperparameters to use for training. Defaults to None.

    Returns
    -------
    None
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

        for epoch in range(100):
            trainer.step()

            wandb.log(
                {
                    "loss": trainer.losses[-1],
                },
            )

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {trainer.losses[-1]}")


def main():
    """Train a feed-forward network to learn a simple function."""
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
                    5,
                    10,
                ],
            },
            "layer_2": {
                "values": [
                    5,
                    10,
                ],
            },
            "epochs": {
                "value": 100,
            },
        },
    }

    # Create a new sweep
    sweep_id = wandb.sweep(sweep_config, project="train-ff")

    # Start the sweep
    wandb.agent(sweep_id, train, count=100)


if __name__ == "__main__":
    main()
