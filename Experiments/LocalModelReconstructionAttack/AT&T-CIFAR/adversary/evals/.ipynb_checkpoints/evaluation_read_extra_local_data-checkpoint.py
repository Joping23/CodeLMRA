import json
import numpy as np
import os
from scipy.special import softmax


def load_extra_local_data(worker_id, data_directory, dim=50, num_samples=10000, sigma=0.1, num_classes=2):
    np.random.seed(0)
    with open(data_directory + '/all_data.json', 'rb') as f:
        data = json.load(f)

    w = data['user_data'][str(worker_id)]['w']

    Sigma = np.zeros((dim, dim))
    for i in range(dim):
        Sigma[i, i] = (i + 1) ** 2

    # Generate x
    B = np.random.normal(loc=0.0, scale=1.0, size=None)
    loc = np.random.normal(loc=B, scale=1.0, size=dim)
    samples = np.ones((num_samples, dim + 1))
    samples[:, 1:] = np.random.multivariate_normal(mean=loc, cov=Sigma, size=num_samples)
    x = samples[:, 1:]

    # Generate y
    
    y = np.matmul(samples, w) + np.random.normal(loc=0., scale=sigma, size=(num_samples, num_classes))

    seperate = int(num_samples / 2)

    filepath: str = os.path.join(data_directory, "train", str(worker_id) + '.json')
    with open(filepath, 'rb') as f:
        data_worker = json.load(f)
    train_x = np.concatenate((x[:seperate], data_worker['x']), axis=0)
    train_y = np.concatenate((y[:seperate], data_worker['y']), axis=0)
    return (train_x, train_y), (x[seperate:], y[seperate:])


def load_optimum_model_to_vector(worker_id, data_directory):
    with open(data_directory+'/all_data.json', 'rb') as f:
        data = json.load(f)
    model = data['user_data'][str(worker_id)]['w']
    model = np.array(model)
    first_layer = model[1:,:]
    second_layer = model[0,:]
    c = first_layer.transpose()
    d = c.reshape(c.shape[0]*c.shape[1])
    final_vector = np.append(d, second_layer)
    return final_vector
