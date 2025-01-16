"""app-pytorch: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
# from flwr_datasets import FederatedDataset # NOTE: flwr_dataset will create ~/.flwr/source
# from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# fds = None  # Cache FederatedDataset


# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader

class DummyDataset(Dataset):
    def __init__(self, num_samples, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.data = torch.randn(num_samples, 3, 32, 32)  # Random images
        self.targets = torch.randint(0, 10, (num_samples,))  # Random labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {'img': self.data[idx], 'label': self.targets[idx]}
        if self.transform:
            sample['img'] = self.transform(sample['img'])
        return sample

def load_data(partition_id: int, num_partitions: int):
    """Load partition dummy CIFAR10 data."""
    num_samples = 50000 // num_partitions  # Assuming 50,000 samples in total
    num_train_samples = int(num_samples * 0.8)
    num_test_samples = num_samples - num_train_samples

    pytorch_transforms = Compose(
        [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = DummyDataset(num_train_samples, transform=pytorch_transforms)
    test_dataset = DummyDataset(num_test_samples, transform=pytorch_transforms)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
