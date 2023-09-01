import os
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder


class AdultDatasetNew():
    def __init__(self,args):
        self.filepath = os.path.join("data_adult_new", "adult_new.csv")
        self.num_workers = args.num_workers
        data = pd.read_csv(self.filepath)

        data.drop(['native-country','marital-status', 'relationship', 'race', 'gender', 'occupation','workclass','education-num','capital-gain', 'capital-loss'], axis=1, inplace=True)
        print(list(data.columns))
        num_samples = 20000
        data = data.sample(num_samples)
        #data['income'] = np.log(data['income']) # because the target distri is left skewed 
        plt.hist(np.log(data['income']), bins = 100)
        plt.savefig("histo_income_log.png")
        self.data = data
        self.seed = args.seed
        self.args = args

    def pre_process(self):
        scaler = MinMaxScaler()
       
        #numerical  = ['age', 'hours-per-week']
        features_log_minmax_transform = pd.DataFrame(data = self.data)
        #dropping the NaN rows now 
        features_log_minmax_transform['age'] = pd.to_numeric(features_log_minmax_transform['age'], downcast="float")
        features_log_minmax_transform['hours-per-week'] = pd.to_numeric(features_log_minmax_transform['hours-per-week'], downcast="float")
        encoder_ed = TargetEncoder()
        categorical = ['education']

        features_log_minmax_transform = features_log_minmax_transform.dropna()
        features_all = features_log_minmax_transform
        print(features_all)
        self.features_final = features_all.loc[(features_all['education'] == 'Doctorate') | (features_all['education'] == 'Masters') |(features_all['education'] == 'Bachelors')]
        self.features_user_1 = self.features_final.loc[(self.features_final['education'] == 'Doctorate')& (self.features_final['age']<38)] # less than 38 years old
        self.features_user_2 = self.features_final.loc[(self.features_final['education'] == 'Doctorate')& (self.features_final['age']<54) & (self.features_final['age']>=38)]
        self.features_user_3 =  self.features_final.loc[(self.features_final['education'] == 'Doctorate')& (self.features_final['age']>53)] #larger than or equal to 53 years old

        self.features_user_1['education'] = encoder_ed.fit_transform(self.features_user_1['education'], self.features_user_1['income'])   
        self.features_user_2['education'] = encoder_ed.fit_transform(self.features_user_2['education'], self.features_user_2['income'])   
        self.features_user_3['education'] = encoder_ed.fit_transform(self.features_user_3['education'], self.features_user_3['income'])   
        self.features_user_1[categorical] = scaler.fit_transform(self.features_user_1[categorical])
        self.features_user_3[categorical] = scaler.fit_transform(self.features_user_3[categorical])
        self.features_user_2[categorical] = scaler.fit_transform(self.features_user_2[categorical])

        self.df_y_user_1 = pd.DataFrame(self.features_user_1.income)
        self.features_user_1 = self.features_user_1.drop(columns=['income'])
        self.df_y_user_2 = pd.DataFrame(self.features_user_2.income)
        self.features_user_2 = self.features_user_2.drop(columns=['income'])
        self.df_y_user_3 = pd.DataFrame(self.features_user_3.income)
        self.features_user_3 = self.features_user_3.drop(columns=['income'])

        self.features_rest = features_all.loc[(features_all['education'] != 'Doctorate')]
        self.features_rest['education'] = encoder_ed.fit_transform(self.features_rest[categorical], self.features_rest['income'])   
        self.features_rest[categorical] = scaler.fit_transform(self.features_rest[categorical])
        print(self.features_rest)
        self.save_all_to_json()


    def save_all_to_json(self):
        user_data = {}
        #num_classes = 2
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

        json_dir = os.path.join("data_adult_new", "all_data.json")

        json_file = {
            'users': [str(i) for i in range(self.num_workers)],
            'num_samples': num_samples,
            'user_data': user_data,
        }

        with open(json_dir, 'w') as outfile:
            json.dump(json_file, outfile)


    def split_data(self):
        data_file = os.path.join('data_adult_new', 'all_data.json')

        with open(data_file, 'r') as inf:
            data = json.load(inf)

        X_list = {"train": [], "test": []}
        y_list = {"train": [], "test": []}


        train_dir = "data_adult_new/train"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        test_dir = "data_adult_new/test"
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

            file = os.path.join("data_adult_new", key, "{}.json".format(key))
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
    data = AdultDatasetNew(args)
    data.pre_process()
    data.split_data()