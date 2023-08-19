import torch

class Tester:
    def __init__(self, model, testloader, device):
        self.model = model
        self.testloader = testloader
        self.device = device

    def test(step):
        total = 0
        correct = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()