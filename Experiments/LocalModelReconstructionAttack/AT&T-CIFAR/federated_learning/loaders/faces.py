import torch
import json
from torch.utils.data import Dataset, DataLoader



class FacesDataset(Dataset):
    def __init__(self, json_file, device):
        self.device = device

        with open(json_file, "r") as f:
            data = json.load(f)

        self.X = torch.tensor(data["x"]).to(device)
        self.y = torch.tensor(data["y"]).to(device)

        self.num_classes = 41
        self.dimension = 80*70

    def __len__(self):
        #print(self.X.shape)
        return self.X.shape[0]

    def __getitem__(self, idx):
        

        return self.X[idx], torch.unsqueeze(self.y[idx], 0)

def get_iterator_faces(file_path, device, batch_size=179):
    """

    :param file_path:
    :param device:
    :param batch_size
    :return:
    """
    dataset = FacesDataset(file_path, device)
    iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return iterator




