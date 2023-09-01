import os
import pandas as pd
import operator
import json
import numpy as np
import random
import argparse
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

class PurchaseDataset():
    """
    def __init__(self):
        self.filepath_x = os.path.join("data_purchase_100", "purchase_100_features.p")
        self.filepath_y = os.path.join("data_purchase_100", "purchase_100_labels.p")

        with open(self.filepath_x, 'rb') as f:
            self.data_x = pd.read_pickle(f)
        with open(self.filepath_y, 'rb') as f:
            self.data_y = pd.read_pickle(f)
        print(self.data_x.shape)
        print(np.unique(self.data_y))

    """
    def __init__(self, num_classes=10, random_seed=1234):
        self.chains = {}
        self.products= {}
        self.num_chosen_products = 100
        self.filename = os.path.join('data_purchase_100', 'purchase_100_raw_data.json')
        self.purchase_100_dict={'info':"seperated_by_chains_with_top_100_popular_products",
                                'chain_names':None,
                                'num_chosen_products':self.num_chosen_products,
                                'selected_product_ids':None,
                                'details':None}
        self.num_all_customers = 0
        self.num_samples=[]
        self.features = []
        self.seed = random_seed
        self.num_classes = num_classes
        if not os.path.exists(self.filename):
            self.read_from_csv()
        with open(self.filename, 'rb') as f:
            self.purchase_100_dict=json.load(f)
        self.chains = self.purchase_100_dict['details']
        self.selected_product_ids = self.purchase_100_dict['selected_product_ids']
        self.num_chosen_products = self.purchase_100_dict['num_chosen_products']
        delete_chains = []
        for chain_id, custom_infor in self.chains.items():
            if len(custom_infor)<100:
                delete_chains.append(chain_id)
        for chain_id in delete_chains:
            del self.chains[chain_id]
        for chain_id, custom_infor in self.chains.items():
            self.num_all_customers+=len(custom_infor)
            chain_customer_features = []
            self.num_samples.append(len(custom_infor))
            for _, products_id in custom_infor.items():
                products_features=[ 1 if i in products_id else 0 for i in self.selected_product_ids]
                chain_customer_features.append(products_features)
            self.features.append(chain_customer_features)
        user_data = {}

        features_2d = np.array([elem for twod in self.features for elem in twod])
        mods = np.linalg.norm(features_2d, axis=1)
        chain_features_array = features_2d / mods[:, np.newaxis]
        y_features_2d = KMeans(n_clusters=self.num_classes, random_state=0).fit(chain_features_array)
        y_features_2d = y_features_2d.labels_
        end = 0
        for ind, chain_features in enumerate(self.features):
            start = end
            number_customers = len(chain_features)
            end = start + number_customers
            chain_features_array = np.array(chain_features)
            mods = np.linalg.norm(chain_features_array, axis=1)
            chain_features_array =  chain_features_array / mods[:, np.newaxis]
            #y_features = KMeans(n_clusters=self.num_classes, random_state=0).fit(chain_features_array)
            #y_features = y_features.labels_.tolist()
            y_features = y_features_2d[start:end].tolist()
            user_data[str(ind)] = {'x':chain_features_array.tolist(), 'y': y_features, 'w': None, 'cluster_id':str(ind)}
        json_dir = os.path.join("data_purchase_100", "all_data.json")
        json_file = {
            'users': [str(i) for i in range(len(self.features))],
            'num_samples': self.num_samples,
            'user_data': user_data,
            "num_classes": self.num_classes
        }
        with open(json_dir, 'w') as outfile:
            json.dump(json_file, outfile)

    def split_data(self):
        data_file = os.path.join('data_purchase_100', 'all_data.json')

        with open(data_file, 'r') as inf:
            data = json.load(inf)

        X_list = {"train": [], "test": []}
        y_list = {"train": [], "test": []}

        num_classes = data['num_classes']

        train_dir = "data_purchase_100/train"
        test_dir = "data_purchase_100/test"
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
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

            file = os.path.join("data_purchase_100", key, "{}.json".format(key))
            json_data = {"x": X.tolist(), "y": y.tolist(), "num_classes": num_classes}
            with open(file, 'w') as outfile:
                json.dump(json_data, outfile)

    def read_from_csv(self):
        file_path = os.path.join("data_purchase_100", "transactions.csv")
        reader = pd.read_csv(file_path, chunksize=100000)
        chain_names = set()
        """
        print("------------------Reading all the chains----------------------")
        all_chains = set()
        for ind, chunk in enumerate(reader):
            set_chunk = set(chunk['chain'].unique())
            all_chains.update(set_chunk)
            print(set_chunk)
            if ind%10 ==0:
                print("---------------------------------------------")
                print(all_chains)
        print(len(all_chains))
        """
        print("------- Reading all the data from csv----------------------")
        for ind, chunk in enumerate(reader):
            for i, info in chunk.iterrows():
                if info['chain'] not in self.chains.keys():
                    self.chains[info['chain']] = {info['id']:set([info['category']])}
                else:
                    if info['id'] not in self.chains[info['chain']].keys():
                        self.chains[info['chain']].update({info['id']:set([info['category']])})
                    else:
                        self.chains[info['chain']][info['id']].add(info['category'])
                if info['category'] not in self.products.keys():
                    self.products[info['category']] = 1
                else:
                    self.products[info['category']] += 1
            chain_names.update(chunk['chain'].values)
            count = 0
            for chain_id, custom_infor in self.chains.items():
                if len(custom_infor)>100:
                    count += 1
            print(f"Chunk:{ind} chain_nums:{len(chain_names)}. Number of chains with more than 100 customers: {count}")
            if count>=10:
                break

        print("------- Finding the top 100 popular products----------------------")
        sorted_products = sorted(self.products.items(), key=operator.itemgetter(1), reverse=True)
        chosen_products = []
        for ind, (id_product, _) in enumerate(sorted_products):
            chosen_products.append(id_product)
            if ind>=self.num_chosen_products-1:
                break
        chosen_products = set(chosen_products)
        print("------- Picking for each user among the top 100 popular products-------------")
        print("Chain_id/ User_id/ Num_products/ Chosen_products")
        for chain_id, info_chain in self.chains.items():
            delete_users = []
            for user_id, set_products in info_chain.items():
                user_chosen_products = set_products.intersection(chosen_products)
                if len(user_chosen_products)>0:
                    info_chain[user_id]= list(user_chosen_products)
                    print(chain_id,user_id,len(set_products),len(info_chain[user_id]))
                else:
                    delete_users.append(user_id)
                    print(f"delete user {user_id}")
            for user_id in delete_users:
                del info_chain[user_id]
        self.purchase_100_dict['chain_names'] = list(self.chains.keys())
        self.purchase_100_dict['selected_product_ids'] = list(chosen_products)
        self.purchase_100_dict['details'] = self.chains
        with open(self.filename, 'w') as f:
            json.dump(self.purchase_100_dict, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_classes',
        help='number of classes of customer behaviours',
        type = int,
        default = 10,
        required=False)
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=931231,
        required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = PurchaseDataset(num_classes=args.num_classes,random_seed=args.seed)
    #data.split_data()