import torch
import json
from torch.utils.data import Dataset, DataLoader



class AdultDatasetNew(Dataset):
    def __init__(self, json_file, device):
        self.device = device

        with open(json_file, "r") as f:
            data = json.load(f)

        self.X = torch.tensor(data["x"]).to(device)
        self.y = torch.tensor(data["y"]).to(device)

        self.dimension = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], torch.unsqueeze(self.y[idx], 0)

def get_iterator_adult_new(file_path, device, batch_size=1):
    """

    :param file_path:
    :param device:
    :param batch_size
    :return:
    """
    dataset = AdultDatasetNew(file_path, device)
    iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return iterator

def get_iterator_purchase_new100(file_path, device, batch_size=1):
    """

    :param file_path:
    :param device:
    :param batch_size
    :return:
    """
    dataset = AdultDatasetNew(file_path, device)
    iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return iterator



