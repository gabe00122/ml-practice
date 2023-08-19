import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 300
        self.linear1 = nn.Linear(32*32*3, hidden)
        self.linear2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = torch.reshape(x, (-1, 32*32*3))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
