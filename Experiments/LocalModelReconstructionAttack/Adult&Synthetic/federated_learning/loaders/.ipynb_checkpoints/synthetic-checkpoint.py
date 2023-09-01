import torch
import json
from torch.utils.data import Dataset, DataLoader
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


class SyntheticDataset(Dataset):
    def __init__(self, json_file, device):
        self.device = device

        with open(json_file, "r") as f:
            data = json.load(f)

        self.X = torch.tensor(data["x"]).to(device)
        self.y = torch.tensor(data["y"]).to(device)

        self.num_classes = data["num_classes"]
        self.dimension = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], torch.unsqueeze(self.y[idx], 0)


def get_iterator_synthetic(file_path, device, batch_size=1, dp=False):
    """

    :param file_path:
    :param device:
    :param batch_size
    :return:
    """
    dataset = SyntheticDataset(file_path, device)
    if "train" in file_path:
        if dp:
            sample_rate = min(batch_size / len(dataset), 1)
            iterator = DataLoader(dataset, batch_sampler=UniformWithReplacementSampler(num_samples=len(dataset),
                                                                                     sample_rate=sample_rate))
        else:
            iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    elif "test" in file_path:
        iterator = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    else:
        print("Error: loaders/ in get_iterator_synthetic")

    return iterator
