from federated_learning.loaders.Faces import get_iterator_faces



def get_iterator(name, path, device, batch_size, data=None, target=None, dp=False, input_type="mlp"):
    if name == "faces":
        return get_iterator_faces(path, device, batch_size = batch_size)
    else:
        raise NotImplementedError
