import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, dataloader, epochs):
        writer = SummaryWriter()
        step = 0

        for epoch in range(epochs):
            print("Starting epoch " + str(epoch))
            
            for i, data in enumerate(dataloader, 0):
                input_data, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(input_data)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                writer.add_scalar('Loss/train', loss.item(), step)
                step += 1

        writer.close()

