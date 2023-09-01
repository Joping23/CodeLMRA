from .model import LinearNet, LinearNetMulti


def get_gradient_model(model, input_size, num_features):
    if model == "nn_linear":
        return LinearNet(input_size, num_features)
    if model == "nn_multiple_linear":
        return LinearNetMulti(input_size)
    else:
        raise NotImplementedError