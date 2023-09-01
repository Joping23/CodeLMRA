import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.optim import get_lr_scheduler, get_optimizer
from .model import Model



class NeuralLayer(nn.Module):
    def __init__(self, input_dimension, intermediate_dimension, num_classes):
        super(NeuralLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.intermediate_dimension = intermediate_dimension
        self.fc1 = nn.Linear(input_dimension, intermediate_dimension)
        self.fc2 = nn.Linear(intermediate_dimension, intermediate_dimension)
        self.fc3 = nn.Linear(intermediate_dimension, intermediate_dimension)

        self.fc4 =  nn.Linear(intermediate_dimension, num_classes)
        self.bn_256 = nn.BatchNorm1d(intermediate_dimension)

        self.relu = nn.ReLU()
    def forward(self, x):
        first_layer = self.relu(self.fc1(x))
        second_layer = self.relu(self.fc2(first_layer))
        third_layer = self.relu(self.fc3(second_layer))
        output = self.fc4(third_layer)
        return output

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()




class NeuralNetworkReg(Model):
    def __init__(self, criterion, metric, device, input_dimension, num_classes, iterator, args,
                 optimizer_name="sgd", lr_scheduler="constant", initial_lr=1e-3, epoch_size=1,coeff=1):
        super(NeuralNetworkReg, self).__init__()

        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.coeff = coeff

        self.net = NeuralLayer(input_dimension, 256, num_classes).to(self.device)

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()

            y = y.view(-1)

            predictions = self.net(x).view(-1)

            loss = self.coeff * self.criterion(predictions, y)

            mape = self.metric(predictions, y)

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += mape.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        self.optimizer.zero_grad()

        y = torch.tensor(y, device=self.device)
        y = y.view(-1)

        predictions = self.net(x).view(-1)

        loss = self.coeff * self.criterion(predictions, y)

        acc = self.metric(predictions, y)

        loss.backward()
        if update:
            self.optimizer.step()
            self.lr_scheduler.step()

        batch_loss = loss.item()
        batch_acc = acc.item()

        return batch_loss, batch_acc

    def evaluate_iterator(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.eval()
        total = 0
        with torch.no_grad():
            for x, y in iterator:
                predictions = self.net(x).view(-1)

                y = y.view(-1)

                loss = self.criterion(predictions, y)

                acc = self.metric(predictions, y)

                epoch_loss += loss.item()*len(y)
                epoch_acc += acc.item()*len(y)
                total += len(y)

        return epoch_loss / total, epoch_acc / total
