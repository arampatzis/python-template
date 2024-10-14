#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset

from template.trainer import FeedForward
from template.trainer import Trainer

def main():
    
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = x ** 2

    trainer = Trainer(
        FeedForward(),
        TensorDataset(x, y),
        batch_size=16
    )
    
    for epoch in range(100):
        
        trainer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {trainer.losses[-1]}')

if __name__ == '__main__':
    main()