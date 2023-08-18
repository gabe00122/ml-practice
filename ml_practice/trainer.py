import torch;
import torch.nn as nn

class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader = torch.utils.data.Data

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            running_loss = 0

            for i, data in enumerate(dataloader, 0):
                input_data, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(input_data)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.8f}')
                running_loss = 0.0
