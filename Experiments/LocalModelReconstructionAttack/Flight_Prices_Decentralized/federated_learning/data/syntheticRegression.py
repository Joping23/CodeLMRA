import os
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_regression,make_low_rank_matrix
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder


class syntheticRegression():
    def __init__(self,args):
        self.num_workers = args.num_workers
        data = make_regression(n_samples=10000,n_features=20,random_state=42)
        

        self.data = data
        self.seed = args.seed
        self.args = args

    def pre_process(self):
     
        self.save_all_to_json()


    def save_all_to_json(self):
        print(self.data[0].shape)
        user_data = {}
        #num_classes = 2
        list_num_samples = []
        num_samples = int(len(self.data[0])/(self.num_workers))
        for i in range(self.num_workers):
            list_num_samples.append(num_samples)

        #Rest workers:
        y = self.data[1].tolist()
        self.features = self.data[0]
        x = self.features.tolist()
        slice_end = 0
        for i in range(0,self.num_workers):
            slice_start = slice_end
            slice_end = slice_start + num_samples
            x_train_sub = x[slice_start:slice_end]
            y_train_sub = y[slice_start:slice_end]
            user_data[str(i)] = {'x': x_train_sub, 'y': y_train_sub, 'w': None, 'cluster_id': i}
            #print(str(i), len(user_data[str(i)]['y']),len(user_data[str(i)]['x']), len(user_data[str(i)]['x'][0]))

        json_dir = os.path.join("data_synthetic_reg_half_rank", "all_data.json")

        json_file = {
            'users': [str(i) for i in range(self.num_workers)],
            'num_samples': num_samples,
            'user_data': user_data,
        }

        with open(json_dir, 'w') as outfile:
            json.dump(json_file, outfile)


    def split_data(self):
        data_file = os.path.join('data_synthetic_reg_half_rank', 'all_data.json')

        with open(data_file, 'r') as inf:
            data = json.load(inf)

        X_list = {"train": [], "test": []}
        y_list = {"train": [], "test": []}


        train_dir = "data_synthetic_reg_half_rank/train"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        test_dir = "data_synthetic_reg_half_rank/test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for worker in data['users']:
            train_file = os.path.join(train_dir, "{}.json".format(worker))
            test_file = os.path.join(test_dir, "{}.json".format(worker))

            worker_data = data['user_data'][worker]
            X = np.array(worker_data['x'])
            y = np.array(worker_data['y'])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.9, random_state=self.seed)
            print(worker, len(X_train), len(X_test))
            X_list["train"].append(X_train)
            y_list["train"].append(y_train)
            X_list["test"].append(X_test)
            y_list["test"].append(y_test)

            json_data_train = {"x": X_train.tolist(), "y": y_train.tolist()}
            json_data_test = {"x": X_test.tolist(), "y": y_test.tolist()}

            with open(train_file, 'w') as outfile:
                json.dump(json_data_train, outfile)
            with open(test_file, 'w') as outfile:
                json.dump(json_data_test, outfile)


        for key in ["train", "test"]:
            X = np.vstack(X_list[key])
            y = np.concatenate(y_list[key])

            file = os.path.join("data_synthetic_reg_half_rank", key, "{}.json".format(key))
            json_data = {"x": X.tolist(), "y": y.tolist()}
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
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=931231,
        required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data = syntheticRegression(args)
    data.pre_process()
    data.split_data()