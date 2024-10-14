import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class FeedForward(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2)
        )

    def forward(self, x):
        return self.fc(x)


class Trainer():
    
    def __init__(self, model, dataset, batch_size=16, lr=0.01):
        
        self.model = model
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        
        self.losses = []    
        
        wandb.watch(self.model)
    
        
    def step(self):
        epoch_loss = 0
        for batch_x, batch_y in self.loader:
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = nn.MSELoss()(output, batch_y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(self.loader)
        self.losses.append(epoch_loss)

