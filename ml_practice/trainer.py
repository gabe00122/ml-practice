import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_interval = 100

    def train(self, dataloader, epochs):
        writer = SummaryWriter()
        step = 0

        for epoch in range(epochs):
            print("Starting epoch " + str(epoch))

            running_loss = 0

            for i, data in enumerate(dataloader, 0):
                input_data, labels = data
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_data)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % self.log_interval == self.log_interval - 1:
                    writer.add_scalar('Loss/train', loss.item(), step)
                step += 1

        writer.close()
