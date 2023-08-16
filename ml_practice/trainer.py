import torch;
import torch.nn as nn

class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer, batch_size: int):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader = torch.utils.data.Data

    def train():
        pass