import torch
import json
import os
from torch.utils.data import Dataset

def find_best_bz(data_size, batch_size):
    for i in range(1, data_size):
        if data_size%(batch_size+i)==0:
            return batch_size+i
        elif data_size%(batch_size-i)==0:
            return batch_size-i

    return data_size

def get_all_data(input_data_dir, worker_id, fl_lr, device):
    filepath = os.path.join(input_data_dir, "inter" + str(worker_id) + ".json")
    with open(filepath, 'rb') as f:
        data_worker = json.load(f)

    data_batch_x = []
    data_batch_y = []
    for global_model, local_model in data_worker:
        g = [(gm - lm) / fl_lr for gm, lm in zip(global_model, local_model)]
        data_batch_x.append(global_model)
        data_batch_y.append(g)

    x = torch.FloatTensor(data_batch_x).to(device)
    y = torch.FloatTensor(data_batch_y).to(device)

    return x,y

class ModelGradientDataset(Dataset):
    def __init__(self, input_data_dir, worker_id, batch_size, fl_lr, device):
        self.device = device
        filepath = os.path.join(input_data_dir, "inter" + str(worker_id) + ".json")
        with open(filepath, 'rb') as f:
            data_worker = json.load(f)
        data_size = len(data_worker)
        if data_size % batch_size != 0:
            batch_size = find_best_bz(data_size, batch_size)

        data_batch_x = []
        data_batch_y = []
        iter = 0
        for global_model, local_model in data_worker:
            if iter % batch_size == 0:
                data_batch_x.append([])
                data_batch_y.append([])
            g = [(gm - lm) / fl_lr for gm, lm in zip(global_model, local_model)]
            data_batch_x[-1].append(global_model)
            data_batch_y[-1].append(g)
            iter += 1

        self.X = torch.FloatTensor(data_batch_x).to(self.device)
        self.y = torch.FloatTensor(data_batch_y).to(self.device)

    def __len__(self):
        return self.X[0].shape[0]*self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
