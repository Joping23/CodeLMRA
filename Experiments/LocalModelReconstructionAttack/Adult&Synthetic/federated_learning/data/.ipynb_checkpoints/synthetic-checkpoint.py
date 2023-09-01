""" From https://github.com/TalwalkarLab/leaf/blob/master/data/synthetic/"""
import argparse
import json
import os
import numpy as np
import random
import shutil
from scipy.special import softmax
from sklearn.model_selection import train_test_split


class SyntheticDataset:
    def __init__(
            self,
            num_classes=2,
            seed=931231,
            num_dim=10,
            num_clusters=5,
            num_workers = 5,
            sigma = 0.1):

        np.random.seed(seed)
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.num_dim = num_dim
        self.num_clusters = num_clusters
        self.prob_clusters = [float(1 / num_clusters) for i in range(num_clusters)]
        self.side_info_dim = num_clusters
        self.y_sigma = sigma
        self.Q = np.random.normal(
            loc=0.0, scale=1.0, size=(self.num_dim + 1, self.num_classes, self.side_info_dim))

        self.Sigma = np.zeros((self.num_dim, self.num_dim))
        for i in range(self.num_dim):
            self.Sigma[i, i] = (i + 1) ** 2

        self.means = self._generate_clusters()

    def get_task(self, num_samples, worker_id):
        if self.num_clusters == self.num_workers:
            cluster_idx = np.int32(worker_id)
        else:
            cluster_idx = np.random.choice(
                range(self.num_clusters), size=None, replace=True, p=self.prob_clusters)
        new_task = self._generate_task(self.means[cluster_idx], cluster_idx, num_samples)
        return new_task

    def _generate_clusters(self):
        means = []
        for i in range(self.num_clusters):
            loc = np.random.normal(loc=0, scale=1., size=None)
            mu = np.random.normal(loc=loc, scale=1., size=self.side_info_dim)
            means.append(mu)
        return means

    def _generate_x(self, num_samples):
        B = np.random.normal(loc=0.0, scale=1.0, size=None)
        loc = np.random.normal(loc=B, scale=1.0, size=self.num_dim)

        samples = np.ones((num_samples, self.num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=self.Sigma, size=num_samples)

        return samples

    def _generate_y(self, x, cluster_mean):
        model_info = np.random.normal(loc=cluster_mean, scale=0.1, size=cluster_mean.shape)
        w = np.matmul(self.Q, model_info)
        print(w.shape, x.shape)
        num_samples = x.shape[0]
        prob = softmax(np.matmul(x, w) + np.random.normal(loc=0., scale=self.y_sigma, size=(num_samples, self.num_classes)),
                       axis=1)

        y = np.argmax(prob, axis=1)
        return y, w, model_info

    def _generate_task(self, cluster_mean, cluster_id, num_samples):
        x = self._generate_x(num_samples)
        y, w, model_info = self._generate_y(x, cluster_mean)

        x = x[:, 1:]

        return {'x': x, 'y': y, 'w': w, 'model_info': model_info, 'cluster': cluster_id}


def get_num_samples(num_tasks, min_num_samples=20, max_num_samples=1000, strategy="even"):
    if strategy == "lognormal":
        num_samples = np.random.lognormal(3, 1.5, (num_tasks)).astype(int)
        num_samples = [min(s + min_num_samples, max_num_samples)*10 for s in num_samples]
    elif strategy == "even":
        num_samples = [64 for i in range(num_tasks)]
    else:
        raise NotImplementedError
    return num_samples


def to_leaf_format(tasks):
    users, num_samples, user_data = [], [], {}

    for i, t in enumerate(tasks):
        x, y, w, cluster_id = t['x'].tolist(), t['y'].tolist(), t['w'].tolist(), t['cluster'].tolist()
        u_id = str(i)

        users.append(u_id)
        num_samples.append(len(y))
        user_data[u_id] = {'x': x, 'y': y, 'w': w, 'cluster_id': cluster_id}

    return users, num_samples, user_data


def save_json(json_dir, json_name, users, num_samples, user_data, num_classes):
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    json_file = {
        'users': users,
        'num_samples': num_samples,
        'user_data': user_data,
        "num_classes": num_classes
    }

    with open(os.path.join(json_dir, json_name), 'w') as outfile:
        json.dump(json_file, outfile)


def split_data(rng_seed=931231):
    rng = random.Random(rng_seed)
    data_file = os.path.join('data_synthetic', 'all_data.json')

    with open(data_file, 'r') as inf:
        data = json.load(inf)

    X_list = {"train": [], "test": []}
    y_list = {"train": [], "test": []}

    num_classes = data['num_classes']

    train_dir = "data_synthetic/train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    for worker in data['users']:
        train_file = os.path.join(train_dir, "{}.json".format(worker))

        worker_data = data['user_data'][worker]
        X = np.array(worker_data['x'])
        y = np.array(worker_data['y'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=rng_seed)

        X_list["train"].append(X_train)
        y_list["train"].append(y_train)
        X_list["test"].append(X_test)
        y_list["test"].append(y_test)

        json_data_train = {"x": X_train.tolist(), "y": y_train.tolist(), "num_classes": num_classes}

        with open(train_file, 'w') as outfile:
            json.dump(json_data_train, outfile)

    test_dir = "data_synthetic/test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for key in ["train", "test"]:
        X = np.vstack(X_list[key])
        y = np.concatenate(y_list[key])

        file = os.path.join("data_synthetic", key, "{}.json".format(key))
        json_data = {"x": X.tolist(), "y": y.tolist(), "num_classes": num_classes}
        with open(file, 'w') as outfile:
            json.dump(json_data, outfile)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_workers',
        help='number of workers;',
        type=int,
        required=True)
    parser.add_argument(
        '--num_clusters',
        help='number of clusters;',
        type=int,
        required=True)
    parser.add_argument(
        '--num_classes',
        help='number of classes;',
        type=int,
        required=True)
    parser.add_argument(
        '--dimension',
        help='data dimension;',
        type=int,
        required=True)
    parser.add_argument(
        '--sigma',
        help='noise scale;',
        type=float,
        default=0.1,
        required=False)
    parser.add_argument(
        '--strategy',
        help='data split strategy',
        type=str,
        default='even',
        required=False
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=931231,
        required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    local_data_dir = os.path.join('data_synthetic', 'train')
    if os.path.isdir(local_data_dir):
        shutil.rmtree(local_data_dir)
    num_samples = get_num_samples(args.num_workers, strategy=args.strategy)
    dataset = SyntheticDataset(
        num_classes=args.num_classes, num_dim=args.dimension, num_clusters=args.num_clusters, num_workers = args.num_workers, seed=args.seed)
    tasks = [dataset.get_task(s,i) for s,i in zip(num_samples, range(len(num_samples)))]
    users, num_samples, user_data = to_leaf_format(tasks)
    save_json('data_synthetic', 'all_data.json', users, num_samples, user_data, args.num_classes)
    with open('data_synthetic/args.txt', 'w') as f:
        print(vars(args), file=f)
    split_data(args.seed)


if __name__ == '__main__':
    main()
