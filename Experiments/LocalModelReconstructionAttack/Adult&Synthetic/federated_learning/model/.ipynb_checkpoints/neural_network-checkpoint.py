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
    #def forward(self, x):
    #    first_layer = self.relu(self.bn_256(self.fc1(x)))
    #    second_layer = self.relu(self.bn_256(self.fc2(first_layer)))
    #    third_layer = self.relu(self.bn_256(self.fc3(second_layer)))
    #    output = self.fc4(third_layer)
    #    return output

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

        
        
        
class NeuralLayer_batch(nn.Module):
    def __init__(self, input_dimension, intermediate_dimension, num_classes):
        super(NeuralLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.intermediate_dimension = intermediate_dimension
        self.fc1 = nn.Linear(input_dimension, intermediate_dimension)
        self.fc2 = nn.Linear(intermediate_dimension, intermediate_dimension)
        self.fc3 = nn.Linear(intermediate_dimension, intermediate_dimension)
        self.bn_256 = nn.BatchNorm1d(intermediate_dimension)
        self_bn_input = nn.BatchNorm1d(input_dimension)
        
        self.fc4 =  nn.Linear(intermediate_dimension, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        first_layer = self.relu(self.bn_256(self.fc1(x)))
        second_layer = self.relu(self.bn_256(self.fc2(first_layer)))
        third_layer = self.relu(self.bn_256(self.fc3(second_layer)))
        output = self.fc2(second_layer)
        return output

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


class NeuralNetwork(Model):
    def __init__(self, criterion, metric, device, input_dimension, num_classes,
                 optimizer_name="sgd", lr_scheduler="constant", initial_lr=1e-3, epoch_size=1):
        super(NeuralNetwork, self).__init__()

        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.net = NeuralLayer(input_dimension, 256, num_classes).to(self.device)

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()

            y = y.long().view(-1)

            predictions = self.net(x)

            loss = self.criterion(predictions, y)

            acc = self.metric(predictions, y)

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        self.optimizer.zero_grad()

#        y = torch.tensor(y, dtype=torch.long, device=self.device).view(-1)
        y = y.long().view(-1)

        predictions = self.net(x)
        loss = self.criterion(predictions, y)

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
                predictions = self.net(x)

                y = y.long().view(-1)

                loss = self.criterion(predictions, y)

                acc = self.metric(predictions, y)

                epoch_loss += loss.item()*len(y)
                epoch_acc += acc.item()*len(y)
                total += len(y)

        return epoch_loss / total, epoch_acc / total


class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        self.fc3 = nn.Linear(32, 10)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc3.reset_parameters()


class ConvNetwork(Model):
    def __init__(self, criterion, metric, device,
                 optimizer_name="sgd", lr_scheduler="constant", initial_lr=1e-3, epoch_size=1):
        super(ConvNetwork, self).__init__()

        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.net = ConvLayer().to(self.device)

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()

            predictions = self.net(x)

            loss = self.criterion(predictions, y)

            acc = self.metric(predictions, y)

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        self.optimizer.zero_grad()

        predictions = self.net(x)
        loss = self.criterion(predictions, y)

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
                predictions = self.net(x)
                loss = self.criterion(predictions, y)

                acc = self.metric(predictions, y)

                epoch_loss += loss.item()*len(y)
                epoch_acc += acc.item()*len(y)
                total += len(y)

        return epoch_loss / total, epoch_acc / total
