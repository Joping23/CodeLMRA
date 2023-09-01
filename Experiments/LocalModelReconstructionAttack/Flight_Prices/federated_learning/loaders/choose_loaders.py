from federated_learning.loaders.sythetic import get_iterator_synthetic
from federated_learning.loaders.adult import get_iterator_adult, get_iterator_purchase100
from federated_learning.loaders.adult_new import get_iterator_adult_new
from federated_learning.loaders.syntheticRegression import get_iterator_synthetic_reg
from federated_learning.loaders.cifar10 import get_iterator_cifar10
from federated_learning.loaders.flightPrices import get_iterator_flight

def get_iterator(name, path, device, batch_size, data=None, target=None, dp=False, input_type="mlp"):
    if name == "synthetic":
        return get_iterator_synthetic(path, device, batch_size=batch_size, dp=dp)
    if name == "flightPrices":
        return get_iterator_flight(path, device, batch_size=batch_size)
    elif name == "adult":
        return get_iterator_adult(path, device, batch_size = batch_size)
    elif name == "synthetic_reg":
        return get_iterator_synthetic_reg(path, device, batch_size = batch_size)
    elif name == "adult_new":
        return get_iterator_adult_new(path, device, batch_size = batch_size)
    elif name == "purchase_100":
        return get_iterator_purchase100(path, device, batch_size = batch_size)
    elif name == "cifar10":
        return get_iterator_cifar10(path, device, cifar10_data=data, cifar10_targets=target, batch_size = batch_size, input_type=input_type)
    else:
        raise NotImplementedError
