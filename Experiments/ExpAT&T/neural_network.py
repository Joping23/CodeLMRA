#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:55:32 2023

@author: idriouich
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import torch
import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit_iterator_one_epoch(self, iterator):
        pass

    @abstractmethod
    def fit_batch(self, iterator):
        pass

    @abstractmethod
    def evaluate_iterator(self, iterator):
        pass

    def update_from_model(self, model):
        """
        update parameters using gradients from another model
        :param model: Model() object, gradients should be precomputed;
        """
        for param_idx, param in enumerate(self.net.parameters()):
            param.grad = list(model.net.parameters())[param_idx].grad.data.clone()

        self.optimizer.step()
        self.lr_scheduler.step()

    def fit_batches(self, iterator, n_steps):
        global_loss = 0
        global_acc = 0

        for step in range(n_steps):
            batch_loss, batch_acc = self.fit_batch(iterator)
            global_loss += batch_loss
            global_acc += batch_acc

        return global_loss / n_steps, global_acc / n_steps

    def fit_iterator(self, train_iterator, val_iterator=None, n_epochs=1, path=None, verbose=0):
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):

            start_time = time.time()

            train_loss, train_acc = self.fit_iterator_one_epoch(train_iterator)
            if val_iterator:
                valid_loss, valid_acc = self.evaluate_iterator(val_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if val_iterator:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if path:
                        torch.save(self.net, path)

            if verbose:
                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Metric: {train_acc * 100:.2f}%')
                if val_iterator:
                    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Metric: {valid_acc * 100:.2f}%')

    def get_param_tensor(self):
        param_list = []

        for param in self.net.parameters():
            param_list.append(param.data.double().view(-1))
        return torch.cat(param_list)




class NeuralLayer(nn.Module):
    def __init__(self, input_dimension, intermediate_dimension, num_classes):
        super(NeuralLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.intermediate_dimension = intermediate_dimension
        self.fc1 = nn.Linear(input_dimension, intermediate_dimension)
        self.fc2 = nn.Linear(intermediate_dimension, num_classes)
        self.relu = nn.ReLU()   
        
    def forward(self, x):
        #print(self.fc1)
        first_layer = self.fc1(x)
        first_layer = self.relu(first_layer)
        output = self.fc2(first_layer)
        return output

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    

class LinearNetMulti(nn.Module):
    def __init__(self, input_dimension, intermediate_dimension, num_classes):
        super(LinearNetMulti, self).__init__()
        self.linearlayer = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, intermediate_dimension),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(),
            #torch.nn.Dropout(0.25),
            torch.nn.Linear(intermediate_dimension, intermediate_dimension),
            torch.nn.ReLU(),
           # torch.nn.Dropout(0.25),
            torch.nn.Linear(intermediate_dimension, num_classes),
        )

    def forward(self, x):
        out = self.linearlayer(x)
        return out
    def reset_parameters(self):
        self.linearlayer.reset_parameters()


class NeuralNetwork(Model):
    def __init__(self, criterion, metric, device, input_dimension, num_classes,
                 optimizer_name="sgd", lr_scheduler="constant", initial_lr=1e-3, epoch_size=1):
        super(NeuralNetwork, self).__init__()

        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.net = NeuralLayer(input_dimension, 32, num_classes).to(self.device)

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            x = x.view(-1, 70*80)
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
        x = x.view(-1, 70*80)
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
                x = x.view(-1, 70*80)
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
        self.conv1 = nn.Conv2d(1, 8, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 4)
        self.conv3 = nn.Conv2d(16, 32, 4)
        self.conv4 = nn.Conv2d(32, 64, 4)
        self.conv5 = nn.Conv2d(64, 128, 4)

        self.fc1 = nn.Linear(3840, 64)
        self.fc3 = nn.Linear(64, 41)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
#        x=self.pool(x)

        x = F.relu(self.conv3(x))
        x=self.pool(x)
        x= F.relu(self.conv4(x))
        x=self.pool(x)

        x= F.relu(self.conv5(x))

        x=self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        
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