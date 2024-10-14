#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import wandb
from torch.utils.data import TensorDataset

from template.trainer import FeedForward
from template.trainer import Trainer

def main():
    
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = x ** 2
    
    config = {
        "lr": 0.01,
        "batch_size": 4,
        "layer_1":  5,
        "layer_2": 5,
    }
    
    wandb.init(
        project="train-ff",
        config=config
    )
    
    trainer = Trainer(
        FeedForward(config["layer_1"], config["layer_2"]),
        TensorDataset(x, y),
        batch_size=config["batch_size"],
        lr=config["lr"],
    )
    
    for epoch in range(100):
        
        trainer.step()
        
        wandb.log({
            "loss": trainer.losses[-1],
        })

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {trainer.losses[-1]}')

if __name__ == '__main__':
    main()