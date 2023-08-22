import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from cifar10 import Cifar10


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, data: Cifar10, device: str) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data = data
        self.device = device

        self._data_iter = iter(self.data.trainloader)

    def _next_data(self) -> tuple:
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data.trainloader)
            return next(self._data_iter)

    def train(self) -> float:
        inputs, labels = self._next_data()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(inputs)

        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()
