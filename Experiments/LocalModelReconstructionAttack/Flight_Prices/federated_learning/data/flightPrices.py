import os
import argparse
import pandas as pd
pd.set_option('display.max_columns', 37)
pd.set_option('display.max_rows', 10)
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class flightPrices():
    def __init__(self,args):
        self.filepath = os.path.join("data_flightPrices", "flightPrices10Airlines.csv")
        self.num_workers = args.num_workers
        data = pd.read_csv(self.filepath)


        data = data.drop('Unnamed: 0',axis=1)
        data = data.drop('Unnamed: 0.1',axis=1)
        data = data.drop('hour_d',axis=1)
        print(list(data.columns))
        self.data = data
        self.seed = args.seed
        self.args = args

    def pre_process(self):
        scaler = MinMaxScaler()
        numerical  = ['trip_duration', 'week_d', 'day_of_week_d','tb_departure']
        features_log_minmax_transform = pd.DataFrame(data = self.data)
        print(features_log_minmax_transform)
        features_log_minmax_transform[numerical] = scaler.fit_transform(self.data[numerical])
        features_all = features_log_minmax_transform  
        encoder_od = TargetEncoder()
        encoder_bc = TargetEncoder()
        #encoder_airline = TargetEncoder()
        scaler_cat = MinMaxScaler()
        categorical = ['od','booking_class']

        



        
        self.features_final = features_all
        print(features_all.columns)
        #categorical_features = ['od','booking_class','op_carrier']
        self.features_user_1 = self.features_final.loc[(self.features_final['op_carrier'] == 'SK')]
        self.features_user_1['booking_class'] = encoder_bc.fit_transform(self.features_user_1['booking_class'], self.features_user_1['price'])   
        self.features_user_1 = self.features_user_1.drop('op_carrier',axis=1)
        self.features_user_1['od'] = encoder_od.fit_transform(self.features_user_1['od'], self.features_user_1['price'])  
        self.features_user_1[categorical] = scaler_cat.fit_transform(self.features_user_1[categorical])

        self.features_user_2 = self.features_final.loc[(self.features_final['op_carrier'] == 'AF')]
        self.features_user_2['booking_class'] = encoder_bc.fit_transform(self.features_user_2['booking_class'], self.features_user_2['price'])   
        self.features_user_2 = self.features_user_2.drop('op_carrier',axis=1)
        self.features_user_2['od'] = encoder_od.fit_transform(self.features_user_2['od'], self.features_user_2['price'])   
        self.features_user_2[categorical] = scaler_cat.fit_transform(self.features_user_2[categorical])

        self.features_user_3 = self.features_final.loc[(self.features_final['op_carrier'] == 'IB')]
        self.features_user_3['booking_class'] = encoder_bc.fit_transform(self.features_user_3['booking_class'], self.features_user_3['price'])   
        self.features_user_3 = self.features_user_3.drop('op_carrier',axis=1)
        self.features_user_3['od'] = encoder_od.fit_transform(self.features_user_3['od'], self.features_user_3['price'])   
        self.features_user_3[categorical] = scaler_cat.fit_transform(self.features_user_3[categorical])

        self.features_user_4 = self.features_final.loc[(self.features_final['op_carrier'] == 'UX')]
        self.features_user_4['booking_class'] = encoder_bc.fit_transform(self.features_user_4['booking_class'], self.features_user_4['price'])   
        self.features_user_4 = self.features_user_4.drop('op_carrier',axis=1)
        self.features_user_4['od'] = encoder_od.fit_transform(self.features_user_4['od'], self.features_user_4['price'])   
        self.features_user_4[categorical] = scaler_cat.fit_transform(self.features_user_4[categorical])

        self.features_user_5 = self.features_final.loc[(self.features_final['op_carrier'] == 'KE')]
        self.features_user_5['booking_class'] = encoder_bc.fit_transform(self.features_user_5['booking_class'], self.features_user_5['price'])   
        self.features_user_5 = self.features_user_5.drop('op_carrier',axis=1)
        self.features_user_5['od'] = encoder_od.fit_transform(self.features_user_5['od'], self.features_user_5['price'])   
        self.features_user_5[categorical] = scaler_cat.fit_transform(self.features_user_5[categorical])

        self.features_user_6 = self.features_final.loc[(self.features_final['op_carrier'] == 'AH')]
        self.features_user_6['booking_class'] = encoder_bc.fit_transform(self.features_user_6['booking_class'], self.features_user_6['price'])   
        self.features_user_6 = self.features_user_6.drop('op_carrier',axis=1)
        self.features_user_6['od'] = encoder_od.fit_transform(self.features_user_6['od'], self.features_user_6['price'])   
        self.features_user_6[categorical] = scaler_cat.fit_transform(self.features_user_6[categorical])

        self.features_user_7 = self.features_final.loc[(self.features_final['op_carrier'] == 'SV')]
        self.features_user_7['booking_class'] = encoder_bc.fit_transform(self.features_user_7['booking_class'], self.features_user_7['price'])   
        self.features_user_7 = self.features_user_7.drop('op_carrier',axis=1)
        self.features_user_7['od'] = encoder_od.fit_transform(self.features_user_7['od'], self.features_user_7['price'])   
        self.features_user_7[categorical] = scaler_cat.fit_transform(self.features_user_7[categorical])

        self.features_user_8 = self.features_final.loc[(self.features_final['op_carrier'] == 'MS')]
        self.features_user_8['booking_class'] = encoder_bc.fit_transform(self.features_user_8['booking_class'], self.features_user_8['price'])   
        self.features_user_8 = self.features_user_8.drop('op_carrier',axis=1)
        self.features_user_8['od'] = encoder_od.fit_transform(self.features_user_8['od'], self.features_user_8['price'])   
        self.features_user_8[categorical] = scaler_cat.fit_transform(self.features_user_8[categorical])

        self.features_user_9 = self.features_final.loc[(self.features_final['op_carrier'] == 'TG')]
        self.features_user_9['booking_class'] = encoder_bc.fit_transform(self.features_user_9['booking_class'], self.features_user_9['price'])   
        self.features_user_9 = self.features_user_9.drop('op_carrier',axis=1)
        self.features_user_9['od'] = encoder_od.fit_transform(self.features_user_9['od'], self.features_user_9['price'])   
        self.features_user_9[categorical] = scaler_cat.fit_transform(self.features_user_9[categorical])

        print(self.features_user_9)
        self.df_y_user_1 = pd.DataFrame(self.features_user_1.price)
        self.features_user_1 = self.features_user_1.drop(columns = ['price'])

        self.df_y_user_2 = pd.DataFrame(self.features_user_2.price)
        self.features_user_2 = self.features_user_2.drop(columns = ['price'])


        self.df_y_user_3 = pd.DataFrame(self.features_user_3.price)
        self.features_user_3 = self.features_user_3.drop(columns = ['price'])

        self.df_y_user_4 = pd.DataFrame(self.features_user_4.price)
        self.features_user_4 = self.features_user_4.drop(columns = ['price'])
    
        
        self.df_y_user_5 = pd.DataFrame(self.features_user_5.price)
        self.features_user_5 = self.features_user_5.drop(columns = ['price'])


        self.df_y_user_6 = pd.DataFrame(self.features_user_6.price)
        self.features_user_6 = self.features_user_6.drop(columns = ['price'])


        self.df_y_user_7 = pd.DataFrame(self.features_user_7.price)
        self.features_user_7 = self.features_user_7.drop(columns = ['price'])


        self.df_y_user_8 = pd.DataFrame(self.features_user_8.price)
        self.features_user_8 = self.features_user_8.drop(columns = ['price'])

        self.df_y_user_9 = pd.DataFrame(self.features_user_9.price)
        self.features_user_9 = self.features_user_9.drop(columns = ['price'])

        self.features_rest = self.features_final.loc[(self.features_final['op_carrier'] == 'LH')]
        self.features_rest['booking_class'] = encoder_bc.fit_transform(self.features_rest['booking_class'], self.features_rest['price'])   
        self.features_rest = self.features_rest.drop('op_carrier',axis=1)
        self.features_rest['od'] = encoder_od.fit_transform(self.features_rest['od'], self.features_rest['price'])   
        self.features_rest[categorical] = scaler_cat.fit_transform(self.features_rest[categorical])

        print(self.features_rest)


        self.save_all_to_json()

    def save_all_to_json(self):
        user_data = {}
        #num_classes = 2
        num_samples = [self.features_user_1.shape[0],self.features_user_2.shape[0], self.features_user_3.shape[0],self.features_user_4.shape[0],self.features_user_5.shape[0],self.features_user_6.shape[0],self.features_user_7.shape[0],self.features_user_8.shape[0],self.features_user_9.shape[0]]
        num_samples_per_rest = int(len(self.features_rest)/(self.num_workers-9))
        for i in range(self.num_workers-9):
            num_samples.append(num_samples_per_rest)

        #First 3 workers:
        user_data['0'] = {'x': self.features_user_1.values.tolist(), 'y': self.df_y_user_1.values.tolist(), 'w': None, 'cluster_id': 0}
        user_data['1'] = {'x': self.features_user_2.values.tolist(), 'y': self.df_y_user_2.values.tolist(), 'w': None,
                          'cluster_id': 1}
        user_data['2'] = {'x': self.features_user_3.values.tolist(), 'y': self.df_y_user_3.values.tolist(), 'w': None,
                          'cluster_id': 2}
        user_data['3'] = {'x': self.features_user_4.values.tolist(), 'y': self.df_y_user_4.values.tolist(), 'w': None,
                          'cluster_id': 3}
        user_data['4'] = {'x': self.features_user_5.values.tolist(), 'y': self.df_y_user_5.values.tolist(), 'w': None,
                          'cluster_id': 4}
        user_data['5'] = {'x': self.features_user_6.values.tolist(), 'y': self.df_y_user_6.values.tolist(), 'w': None,
                          'cluster_id': 5}
        user_data['6'] = {'x': self.features_user_7.values.tolist(), 'y': self.df_y_user_7.values.tolist(), 'w': None,
                          'cluster_id': 6}
        user_data['7'] = {'x': self.features_user_8.values.tolist(), 'y': self.df_y_user_8.values.tolist(), 'w': None,
                          'cluster_id': 7}
        user_data['8'] = {'x': self.features_user_9.values.tolist(), 'y': self.df_y_user_9.values.tolist(), 'w': None,
                          'cluster_id': 8}

        #Rest workers:
        y = pd.DataFrame(self.features_rest.price).values.tolist()
        features = self.features_rest
        features = features.drop(columns = ['price'])
        x = features.values.tolist()
        slice_end = 0
        for i in range(9,self.num_workers):
            slice_start = slice_end
            slice_end = slice_start + num_samples_per_rest
            x_train_sub = x[slice_start:slice_end]
            y_train_sub = y[slice_start:slice_end]
            user_data[str(i)] = {'x': x_train_sub, 'y': y_train_sub, 'w': None, 'cluster_id': i}
            #print(str(i), len(user_data[str(i)]['y']),len(user_data[str(i)]['x']), len(user_data[str(i)]['x'][0]))

        json_dir = os.path.join("data_flightPrices", "all_data.json")

        json_file = {
            'users': [str(i) for i in range(self.num_workers)],
            'num_samples': num_samples,
            'user_data': user_data,
        }

        with open(json_dir, 'w') as outfile:
            json.dump(json_file, outfile)


    def split_data(self):
        data_file = os.path.join('data_flightPrices', 'all_data.json')

        with open(data_file, 'r') as inf:
            data = json.load(inf)

        X_list = {"train": [], "test": []}
        y_list = {"train": [], "test": []}


        train_dir = "data_flightPrices/train"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        test_dir = "data_flightPrices/test"
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

            file = os.path.join("data_flightPrices", key, "{}.json".format(key))
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
    data = flightPrices(args)
    data.pre_process()
    data.split_data()