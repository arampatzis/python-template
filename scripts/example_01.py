#!/usr/bin/env python3

"""Train a feed-forward network to learn a simple function."""


import torch
from torch.utils.data import TensorDataset

import wandb
from template.trainer import FeedForward, Trainer


def main():
    r"""
    Train a feed-forward network to learn the function $f(x) = x^2$ for $x\in[-1,1]$
    using 100 data points.
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

    assert isinstance(config["layer_1"], int)
    assert isinstance(config["layer_2"], int)
    assert isinstance(config["batch_size"], int)

    trainer = Trainer(
        FeedForward(config["layer_1"], config["layer_2"]),
        TensorDataset(x, y),
        batch_size=config["batch_size"],
        lr=config["lr"],
    )

    for epoch in range(200):
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
