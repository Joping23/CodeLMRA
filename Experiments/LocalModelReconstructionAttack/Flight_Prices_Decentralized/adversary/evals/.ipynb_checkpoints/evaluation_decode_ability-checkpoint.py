import torch.nn as nn
import torch
import os
import json
from federated_learning.utils.metric import accuracy, mape
from .get_local_model_structure import get_local_model_structure, map_vector_to_net


def evaluate_decoded_model_ability(adversary_local_model, device, worker_id, data_test, data_directory, model, data_train_iterator=None):
    metric =  nn.MSELoss()
    criterion = nn.MSELoss()
    # customize it for regression and classif
    if model == 'linearRegression' or model == 'NeuralReg' :
        metric =  nn.MSELoss()
        criterion = nn.MSELoss()
    net, num_classes, num_dim = get_local_model_structure(model, data_directory)
    map_vector_to_net(adversary_local_model, net, num_classes, num_dim, model)
    net.to(device)
    if data_train_iterator == None:
        filepath: str = os.path.join(data_directory, "train", str(worker_id) + '.json')
        with open(filepath, 'rb') as f:
            data_train = json.load(f)
        x = torch.FloatTensor(data_train['x']).to(device)
        y = torch.LongTensor(data_train['y']).view(-1).to(device)
        prediction = net(x)
        train_acc = metric(prediction, y)
        print(criterion)
        loss = criterion(prediction, y)
        x = torch.FloatTensor(data_test[0]).to(device)
        y = torch.LongTensor(data_test[1]).view(-1).to(device)
        prediction = net(x)
        test_acc = metric(prediction, y)
    else:
        train_acc = 0
        loss = 0
        total = 0
        for x, y in data_train_iterator:
            if model != "conv":
                y = y.long().view(-1)
            prediction = net(x)
            acc = metric(prediction, y)
            train_loss = criterion(prediction, y)
            train_acc += acc * len(y)
            loss += train_loss * len(y)
            total += len(y)
        train_acc /= total
        loss /= total

        test_acc = 0
        total = 0
        for x, y in data_test:
            if model != "conv":
                y = y.long().view(-1)
            prediction = net(x)
            acc = accuracy(prediction, y)
            test_acc += acc * len(y)
            total += len(y)
        test_acc /= total

    return loss.item(), train_acc.item(), test_acc.item()


def evaluate_decoded_model_ability_from_net(net, device, worker_id, data_test, data_directory, model):
    metric =  nn.MSELoss()
    criterion = nn.MSELoss()
    
    if model == 'linearRegression' or model == 'NeuralReg' :
        metric =  nn.MSELoss()
        criterion = nn.MSELoss()
    filepath: str = os.path.join(data_directory, "train", str(worker_id) + '.json')
    with open(filepath, 'rb') as f:
        data_train = json.load(f)
    x = torch.FloatTensor(data_train['x']).to(device)
    y = torch.LongTensor(data_train['y']).view(-1).to(device)
    prediction = net(x)
    train_acc = metric(prediction, y)
    loss = criterion(prediction, y)
    x = torch.FloatTensor(data_test[0]).to(device)
    y = torch.LongTensor(data_test[1]).view(-1).to(device)
    prediction = net(x)
    test_acc = metric(prediction, y)
    return loss.item(), train_acc.item(), test_acc.item()

def extraction_acc(extract_model, optimum_model, device, data_test, data_directory, model):
    metric =  nn.MSELoss()
    criterion = nn.MSELoss()
    
    if model == 'linearRegression' or model == 'NeuralReg' :
        metric =  nn.MSELoss()
    net_extract, num_classes, num_dim = get_local_model_structure(model, data_directory)
    map_vector_to_net(extract_model, net_extract, num_classes, num_dim, model)
    net_extract.to(device)

    net_optimum, num_classes, num_dim = get_local_model_structure(model, data_directory)
    map_vector_to_net(optimum_model, net_optimum, num_classes, num_dim, model)
    net_optimum.to(device)
    if "cifar" not in data_directory:
        x = torch.FloatTensor(data_test[0]).to(device)
        prediction_extract = net_extract(x)
        prediction_optimum = net_optimum(x)
        _, predicted_optimum = torch.max(torch.softmax(prediction_optimum, dim=1), 1)
        acc = metric(prediction_extract, predicted_optimum)
    else:
        acc = 0
        total = 0
        for x, y in data_test:
            if model != "conv":
                y = y.long().view(-1)
            prediction_extract = net_extract(x)
            prediction_optimum = net_optimum(x)
            _, predicted_optimum = torch.max(torch.softmax(prediction_optimum, dim=1), 1)
            acc += metric(prediction_extract, predicted_optimum) * len(y)
            total += len(y)
        acc /= total

    return acc.item()