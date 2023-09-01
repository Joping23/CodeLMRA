import random
import time
from torch.utils.data import ConcatDataset
import numpy as np
from torchvision import datasets, transforms
import os
import json
import argparse

# load cifar10
# save it in json in data_cifar_10/train,test
# split the data by getting the list of indices for each client
# go back to the data, split it in the directories


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def split_dataset_by_labels(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        _, label = dataset[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

    return clients_indices


def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards

    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices


def downoload_dataset():
    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    CIFAR10_MEAN_ = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV_ = (0.2023, 0.1994, 0.2010)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN_, CIFAR10_STD_DEV_),
    ])

    train_dataset = datasets.CIFAR10(root=cifar10_path, train=True, download=True,
               transform=transform)
    test_dataset = datasets.CIFAR10(root=cifar10_path, train=False, download=True,
               transform=transform)
    return train_dataset, test_dataset

def split_data_json(indices):

    workers_ids = list(range(len(indices)))
    train_dir = "data_cifar10/train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    test_dir = "data_cifar10/test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    X_list = {"train": [], "test": []}

    for worker_id in workers_ids:
        worker_data_train = {}
        worker_data_test = {}

        worker_data_train['indices'] = [i for i in indices[worker_id][:int(len(indices[worker_id]) * 9/10)]]
        worker_data_train["num_classes"] = 10

        worker_data_test['indices'] = [i for i in indices[worker_id][int(len(indices[worker_id]) * 9/10):]]
        worker_data_test["num_classes"] = 10

        train_file = os.path.join(train_dir, "{}.json".format(worker_id))
        test_file = os.path.join(test_dir, "{}.json".format(worker_id))

        with open(train_file, 'w') as outfile:
            json.dump(worker_data_train, outfile)
        with open(test_file, 'w') as outfile:
            json.dump(worker_data_test, outfile)

        print(len(worker_data_train['indices']))

        X_list["train"].extend(worker_data_train['indices'])
        X_list["test"].extend(worker_data_test['indices'])

    for key in ["train", "test"]:
        file = os.path.join("data_cifar10", key, "{}.json".format(key))
        json_data = {"indices": X_list[key], "num_classes": 10}
        with open(file, 'w') as outfile:
            json.dump(json_data, outfile)

    return print("---------- data is splitted across the clients ------------")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_workers',
        help='number of workers;',
        type=int,
        required=True)
    parser.add_argument(
        '--n_classes',
        help='number of classes;',
        type=int,
        default=10,
        required=False)
    parser.add_argument(
        '--n_clusters',
        help='number of clusters;',
        type=int,
        default=2,
        required=False)
    parser.add_argument(
        '--alpha',
        help='parameter controlling the diversity among clients',
        type=float,
        default=0.5,
        required=False)

    return parser.parse_args()

def main():
    args = parse_args()
    train_dataset, test_dataset = downoload_dataset()
    train_test_dataset = ConcatDataset([train_dataset, test_dataset])
    indices = split_dataset_by_labels(train_test_dataset, args.n_classes, args.n_workers, args.n_clusters, args.alpha, 1, seed=1234)
    split_data_json(indices)

if __name__ == '__main__':
    main()

