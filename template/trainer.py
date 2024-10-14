import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.fc(x)


class Trainer():
    
    def __init__(self, model, dataset, batch_size=16):
        
        self.model = model
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        self.losses = []    
    
        
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

