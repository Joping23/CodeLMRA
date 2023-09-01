import os
import torch
import json
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CIFARDataset(Dataset):
    def __init__(self, path, device, cifar10_data=None, cifar10_targets=None, input_type="mlp"):
        self.device = device
        self.input_type = input_type
        self.transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
        ])


        with open(path, "r") as f:
            self.indices = json.load(f)['indices']

            if cifar10_data is None or cifar10_targets is None:
                self.data, self.targets = get_cifar10()
            else:
                self.data, self.targets = cifar10_data, cifar10_targets

            self.data = self.data[self.indices]
            self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy())
        img = self.transformer(img)

        #img = torch.tensor(img, dtype=torch.float32)
        if self.input_type == "mlp":
            img = torch.squeeze(img.view(-1, 32 * 32 * 3))
            target = torch.unsqueeze(target, 0)
        return img.to(self.device), target.to(self.device)

def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded
    :return:
        cifar10_data, cifar10_targets
    """
    cifar10_path = os.path.join("federated_learning", "data", "data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    CIFAR10_MEAN_ = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV_ =  (0.2023, 0.1994, 0.2010)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN_, CIFAR10_STD_DEV_),
    ])


    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, transform=transform, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False, transform=transform,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets

def get_iterator_cifar10(file_path, device, cifar10_data=None, cifar10_targets=None,batch_size=1, input_type="mlp"):
    """

    :param file_path:
    :param device:
    :param batch_size
    :return:
    """
    dataset = CIFARDataset(file_path, device, cifar10_data=cifar10_data, cifar10_targets=cifar10_targets, input_type=input_type)
    iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return iterator