import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from net import Net
from trainer import Trainer
from tester import Tester
from cifar10 import Cifar10

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

compiled = False
batch_size = 4
lr = 0.0004

cifar10 = Cifar10(batch_size)
net = Net().to(device)
if compiled:
    net = torch.compile(net, backend="inductor")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

trainer = Trainer(net, criterion, optimizer, cifar10, device)
tester = Tester(net, cifar10, device)

total_batches = len(cifar10.trainloader)

writer = SummaryWriter()

print(total_batches)

for batch in range(total_batches * 1):
    trainer.train()

    if batch % 1250 == 0:
        step = batch * cifar10.batch_size
        print(f"Step: {step}")

        accuracy = tester.test()
        print(f"Accuracy: {accuracy}")
        writer.add_scalar("Training/Accuracy", accuracy, step)


accuracy = tester.test()
writer.add_scalar("Training/Accuracy", accuracy, total_batches * cifar10.batch_size)

writer.add_hparams(
    hparam_dict={"lr": 0.0004, "batch_size": batch_size},
    metric_dict={"Final/Accuracy": accuracy}
)

writer.close()
