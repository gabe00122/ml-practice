import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_folder = './data'

class Cifar10:
    def __init__(self, batch_size: int = 1) -> None:
        self.batch_size = batch_size

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                     download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                                    download=True, transform=transform)
        
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
        self.testloader = DataLoader(self.testset, batch_size=batch_size,
                                     shuffle=False, num_workers=0)
