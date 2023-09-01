import json
import os
import torch
import numpy as np
from federated_learning.model.neural_network import NeuralLayer, ConvLayer, LinearNetMulti
from federated_learning.model.synthetic import LinearLayer
from adversary.gradient_model.choose_model import get_gradient_model
def get_local_model_structure(model, local_data_dir):
    if model == "neuralReg" :
        filepath = os.path.join(local_data_dir, "train", "0.json")
        with open(filepath, 'rb') as f:
            data = json.load(f)
        num_dimension = torch.tensor(data["x"]).shape[1]
        return LinearLayer(num_dimension, 
                           #20,
                           1), 1, num_dimension
    
    elif model == "linearRegression" :
        filepath = os.path.join(local_data_dir, "train", "0.json")
        with open(filepath, 'rb') as f:
             data = json.load(f)
        num_dimension = torch.tensor(data["x"]).shape[1]
        return LinearLayer(num_dimension,1), 1, num_dimension
    
    elif model == "linear" or model == "neural":
        filepath = os.path.join(local_data_dir, "train", "0.json")
        with open(filepath, 'rb') as f:
            data = json.load(f)

        num_dimension = torch.tensor(data["x"]).shape[1]
        num_classes = data["num_classes"]
        if model == "linear":
            return LinearLayer(80*70, 41), 41, num_dimension
        if model == "neural":
            return NeuralLayer(70*80, 32, 41), 41, num_dimension
    
    elif model == "mlp":
        return NeuralLayer(32*32*3, 50, 10), 10, 32*32*3
    elif model == "conv":
        return ConvLayer(), 10, 32*32*3
    else:
        raise NotImplementedError


def map_vector_to_net(adversary_local_model, net, num_classes, num_dimension, model):
    num_classes=41
    num_dimension=70*80
    if model == "linear" or model == 'linearRegression':
        iter = 0
        for param in net.parameters():
            if iter == 0:
                to_fix = adversary_local_model[:-num_classes]
                # print(to_fix)
                param.data = to_fix.reshape(num_classes, num_dimension)
            else:
                param.data = adversary_local_model[-num_classes:]
            iter += 1
    elif model == "neural" or model == "mlp" or model == "conv" or model =="neuralReg":
        start = 0
        for param in net.parameters():
            size_of_param = np.prod(list(param.data.shape))
            end = start + size_of_param
            to_fix = adversary_local_model[start:end]
            param.data = to_fix.reshape(param.data.shape)
            #print(f"{param.data.size()},{adversary_local_model[start:end].size()}")
            start = end
    else:
        raise NotImplementedError

def freeze_model(net, grad_vector):
    #print(grad_vector.size())
    start = 0
    for param_name in net.state_dict():
        size_of_param = np.prod(list(net.state_dict()[param_name].data.size()))
        end = start + size_of_param
        #print(f"{param_name},{size_of_param}")
        if "conv" in param_name:
            #print(f"zero out {param_name}, {grad_vector[0][start:end].size()}")
            grad_vector[0][start:end] = 0
        start = end
    #update_list = filter(lambda p: p == 0, grad_vector[0])

### here what i need : 
def map_vector_to_gradient_net(adversary_local_model, gradient_network_type, input_size, num_features):
    adversary_local_model = torch.tensor(adversary_local_model)
    net = get_gradient_model(gradient_network_type, input_size, num_features)
    num_to_count = num_features*input_size
    start = 0
    if gradient_network_type == "nn_linear":
        for index, param in enumerate(net.parameters()):
            #print(param.data.shape)
            size_of_param = np.prod(list(param.data.shape))
            end = start + size_of_param
            to_fix = adversary_local_model[start:end]
            param.data = to_fix.reshape(param.data.shape)
            start = end
            """
            if index == 0:
                end = start+num_to_count
                param.data = adversary_local_model[start:end].reshape(num_features, input_size)
                start = end
            elif index == 1 or index == 2 or index == 3:
                end = start+num_features
                param.data = adversary_local_model[start:end]
                start = end
            elif index == 4:
                end = start+num_to_count
                param.data = adversary_local_model[start:end].reshape(input_size, num_features)
                start = end
            elif index == 5:
                param.data = adversary_local_model[start:]
            """
    else:
        raise NotImplementedError
    return net

def map_net_to_vector(model_worker, type="learning_network"):
    if type == "learning_network":
        iter_num = 0
        for param in model_worker.net.parameters():
            # print() to see the size
            if iter_num == 0:
                train_model = param.data.clone().view(-1)
            else:
                train_model = torch.cat((train_model, param.data.clone().view(-1)))
            iter_num += 1
    elif type == "gradient_network":
        iter_num = 0
        for param in model_worker.parameters():
            if iter_num == 0:
                train_model = param.data.clone().view(-1)
            else:
                train_model = torch.cat((train_model, param.data.clone().view(-1)))
            iter_num += 1
    else:
        raise NotImplementedError
    return train_model