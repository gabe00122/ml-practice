import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 100
        self.linear1 = nn.Linear(32*32*3, hidden)
        self.linear2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = torch.reshape(x, (-1, 32*32*3))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)


def train():
    for epoch in range(100):
        running_loss = 0

        for i, data in enumerate(trainloader, 0):
            input_data, labels = data

            # target = torch.empty((10,), dtype=torch.float32)
            # target[labels] = 1.0

            optimizer.zero_grad()

            outputs = net(input_data)
            #daprint(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.8f}')
            running_loss = 0.0

    torch.save(net.state_dict(), 'models/cifar_linear.pth')


def test():
    total = 0
    correct = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


train()

#net.load_state_dict(torch.load('./cifar_linear.pth'))
test()
