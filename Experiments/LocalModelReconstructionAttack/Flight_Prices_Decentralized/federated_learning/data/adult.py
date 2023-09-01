import os
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class AdultDataset():
    def __init__(self,args):
        self.filepath = os.path.join("data_adult", "adult.csv")
        self.num_workers = args.num_workers
        data = pd.read_csv(self.filepath)
        data = data[data["workclass"] != "?"]
        data = data[data["occupation"] != "?"]
        #self.data = data[data["native-country"] != "?"]
        data.drop(['fnlwgt'], axis = 1, inplace=True)
        data.drop(['native-country'], axis=1, inplace=True)


        print(list(data.columns))
        self.data = data
        self.seed = args.seed
        self.args = args

    def pre_process(self):
        scaler = MinMaxScaler()
        numerical  = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        features_log_minmax_transform = pd.DataFrame(data = self.data)
        features_log_minmax_transform[numerical] = scaler.fit_transform(self.data[numerical])
        features_log_minmax_transform['income'] = features_log_minmax_transform['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
        features_all = pd.get_dummies(features_log_minmax_transform)
        self.features_final = features_all.loc[(features_all['education_Doctorate'] == 1) | (features_all['education_Masters'] == 1) |(features_all['education_Bachelors'] == 1)]
        self.features_user_1 = self.features_final.loc[(self.features_final['education_Doctorate'] == 1)& (self.features_final['age']<0.3)] # less than 38 years old
        self.features_user_2 = self.features_final.loc[(self.features_final['education_Doctorate'] == 1)& (self.features_final['age']<0.5) & (self.features_final['age']>=0.3)]
        self.features_user_3 =  self.features_final.loc[(self.features_final['education_Doctorate'] == 1)& (self.features_final['age']>0.5)] #larger than or equal to 53 years old
        self.df_y_user_1 = pd.DataFrame(self.features_user_1.income)
        self.features_user_1 = self.features_user_1.drop(columns=['income'])
        self.df_y_user_2 = pd.DataFrame(self.features_user_2.income)
        self.features_user_2 = self.features_user_2.drop(columns=['income'])
        self.df_y_user_3 = pd.DataFrame(self.features_user_3.income)
        self.features_user_3 = self.features_user_3.drop(columns=['income'])
        print(features_all.columns)

        self.features_rest = self.features_final.loc[(self.features_final['education_Doctorate'] == 0)]
        self.save_all_to_json()


    def save_all_to_json(self):
        user_data = {}
        num_classes = 2
        num_samples = [self.features_user_1.shape[0],self.features_user_2.shape[0], self.features_user_3.shape[0]]
        num_samples_per_rest = int(len(self.features_rest)/(self.num_workers-3))
        for i in range(self.num_workers-3):
            num_samples.append(num_samples_per_rest)

        #First 3 workers:
        user_data['0'] = {'x': self.features_user_1.values.tolist(), 'y': self.df_y_user_1.values.tolist(), 'w': None, 'cluster_id': 0}
        user_data['1'] = {'x': self.features_user_2.values.tolist(), 'y': self.df_y_user_2.values.tolist(), 'w': None,
                          'cluster_id': 1}
        user_data['2'] = {'x': self.features_user_3.values.tolist(), 'y': self.df_y_user_3.values.tolist(), 'w': None,
                          'cluster_id': 2}
        #Rest workers:
        y = pd.DataFrame(self.features_rest.income).values.tolist()
        features = self.features_rest
        features = features.drop(columns = ['income'])
        x = features.values.tolist()
        slice_end = 0
        for i in range(3,self.num_workers):
            slice_start = slice_end
            slice_end = slice_start + num_samples_per_rest
            x_train_sub = x[slice_start:slice_end]
            y_train_sub = y[slice_start:slice_end]
            user_data[str(i)] = {'x': x_train_sub, 'y': y_train_sub, 'w': None, 'cluster_id': i}
            #print(str(i), len(user_data[str(i)]['y']),len(user_data[str(i)]['x']), len(user_data[str(i)]['x'][0]))

        json_dir = os.path.join("data_adult", "all_data.json")

        json_file = {
            'users': [str(i) for i in range(self.num_workers)],
            'num_samples': num_samples,
            'user_data': user_data,
            "num_classes": num_classes
        }

        with open(json_dir, 'w') as outfile:
            json.dump(json_file, outfile)


    def split_data(self):
        data_file = os.path.join('data_adult', 'all_data.json')

        with open(data_file, 'r') as inf:
            data = json.load(inf)

        X_list = {"train": [], "test": []}
        y_list = {"train": [], "test": []}

        num_classes = data['num_classes']

        train_dir = "data_adult/train"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        test_dir = "data_adult/test"
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

            json_data_train = {"x": X_train.tolist(), "y": y_train.tolist(), "num_classes": num_classes}
            json_data_test = {"x": X_test.tolist(), "y": y_test.tolist(), "num_classes": num_classes}

            with open(train_file, 'w') as outfile:
                json.dump(json_data_train, outfile)
            with open(test_file, 'w') as outfile:
                json.dump(json_data_test, outfile)


        for key in ["train", "test"]:
            X = np.vstack(X_list[key])
            y = np.concatenate(y_list[key])

            file = os.path.join("data_adult", key, "{}.json".format(key))
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
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=931231,
        required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data = AdultDataset(args)
    data.pre_process()
    data.split_data()