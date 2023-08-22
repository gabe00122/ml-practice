import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from cifar10 import Cifar10


class Tester:
    def __init__(self, model: nn.Module, data: Cifar10, device: str):
        self.model = model
        self.data = data
        self.device = device

    def test(self) -> float:
        total = 0
        correct = 0

        with torch.no_grad():
            for data in self.data.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total