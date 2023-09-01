import torch
import torch.nn as nn
from .synthetic import LinearModel
from .linearRegressor import LinearRegressor
# import ....
from .neural_network import NeuralNetwork, ConvNetwork
from .neural_network_regression import NeuralNetworkReg
from ..utils.metric import accuracy, mape


def get_model(args, name, device, iterator, epoch_size, optimizer_name="sgd", lr_scheduler="constant",
              initial_lr=1e-1, seed=1234, coeff=1):
    """
    Load Model object corresponding to the experiment
    :param args: the arguments for the experiment
    :param name: experiment name; possible are: synthetic, shakespeare, sent140, inaturalist, femnist
    :param device:
    :param iterator: torch.utils.DataLoader object representing an iterator of dataset corresponding to name
    :param epoch_size:
    :param optimizer_name: optimizer name, for now only "adam" is possible
    :param lr_scheduler:
    :param initial_lr:
    :param seed:
    :param coeff
    :return: Model object
    """

    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if name == "linear":
        input_dimension = iterator.dataset.dimension
        num_classes = iterator.dataset.num_classes
        metric = accuracy
        criterion = nn.CrossEntropyLoss()
        return LinearModel(criterion, metric, device, input_dimension, num_classes, iterator, args,
                           optimizer_name, lr_scheduler, initial_lr, epoch_size,coeff=coeff)
    elif name == "linearRegression":
        input_dimension = iterator.dataset.dimension
        metric = mape
        criterion = nn.MSELoss()
        num_classes = 1
        return LinearRegressor(criterion, metric, device, input_dimension, num_classes, iterator, args,
                            optimizer_name, lr_scheduler, initial_lr, epoch_size,coeff=coeff)
    elif name == "neuralReg":
        input_dimension = iterator.dataset.dimension
        metric = mape
        criterion = nn.MSELoss()
        num_classes = 1
        return NeuralNetworkReg(criterion, metric, device, input_dimension, num_classes, iterator, args,
                            optimizer_name, lr_scheduler, initial_lr, epoch_size,coeff=coeff)


    elif name == "neural":
        input_dimension = iterator.dataset.dimension
        print(input_dimension)
        num_classes = iterator.dataset.num_classes
        metric = accuracy
        criterion = nn.CrossEntropyLoss()
        return NeuralNetwork(criterion, metric, device, input_dimension, num_classes,
                           optimizer_name, lr_scheduler, initial_lr, epoch_size)

    elif name == "mlp":
        metric = accuracy
        criterion = nn.CrossEntropyLoss()
        return NeuralNetwork(criterion, metric, device, 32*32*3, 10,
                           optimizer_name, lr_scheduler, initial_lr, epoch_size)

    elif name == "conv":
        metric = accuracy
        criterion = nn.CrossEntropyLoss()
        return ConvNetwork(criterion, metric, device,
                           optimizer_name, lr_scheduler, initial_lr, epoch_size)

    else:
        raise NotImplementedError