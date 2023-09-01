from adversary.evals.get_local_model_structure import get_local_model_structure, map_vector_to_net
import torch

def return_prediction(model, x, y, experiment, local_data_dir):
    net, num_classes, num_dimension = get_local_model_structure(experiment, local_data_dir)
    map_vector_to_net(torch.tensor(model), net, num_classes, num_dimension, experiment)
    prediction = torch.softmax(net(x), dim=1)
    pred = [i[j].item() for i, j in zip(prediction, y)]
    return pred, net(x).tolist()

def return_loss(model, x, y, experiment, local_data_dir, criterion):
    net, num_classes, num_dimension = get_local_model_structure(experiment, local_data_dir)
    map_vector_to_net(torch.tensor(model), net, num_classes, num_dimension, experiment)
    predictions = net(x)
    loss = criterion(predictions, y)
    return loss.item()
