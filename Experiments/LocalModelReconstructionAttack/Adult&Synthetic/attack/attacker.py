import json
import os
import numpy as np
import torch
from .utils import return_prediction
from adversary.evals.get_local_model_structure import get_local_model_structure, map_vector_to_net
from adversary.evals.evaluation_read_extra_local_data import load_extra_local_data
from adversary.evals.get_local_model_structure import get_local_model_structure
import seaborn as sns
from adversary.evals.evaluation_decode_ability import extraction_acc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lime
import lime.lime_tabular
from lime import submodular_pick



class Insider_attack:
    def __init__(self, args):
        self.num_workers = args.num_workers
        self.args = args
        self.device = args.device
        self.ability = args.adversary_ability
        print(self.ability)
        self.file_name = self.args.experiment+"_w_"+str(self.args.num_workers)+"_lr_"+str(self.args.lr)+"_bz_"+str(self.args.bz)+"_fit_epoch_"+str(self.args.fit_by_epoch)+"_local_step_"+str(self.args.num_local_steps)\
                    +"_start_point_"+self.args.start_point+"_ability_"+self.ability+"_pre_"+str(args.precentage_attack)+"_dp_"+str(self.args.DP)+"_epsilon_"+str(self.args.epsilon)+".json"
        with open(self.file_name, 'rb') as f:
            data = json.load(f)
        self.data = data
        self.decoded_models = data['decoded_model']
        self.server_model = data['server_model']
        self.insider_attack_criteria = 0.1
        #self.init_models = data['initial_models']

        #To make sure all use the same optimum model
        #self.file_name_none = self.args.experiment+"_w_"+str(self.args.num_workers)+"_lr_"+str(self.args.lr)+"_bz_"+str(self.args.bz)+"_fit_epoch_"+str(self.args.fit_by_epoch)+"_local_step_"+str(self.args.num_local_steps)\
        #            +"_start_point_"+self.args.start_point+"_ability_none_pre_0.1.json"
        with open(self.file_name, 'rb') as f:
            data_none = json.load(f)
        print(data["args"])
        self.optimal_models = data_none['optimal_model']
        self.decoded_models_performance = {"extraction_accuracy": [], "train_accuracy": [], "test_accuracy": []}
        self.optimal_models_performance = {"train_accuracy": [], "test_accuracy":[]}
        self.local_data_dir = os.path.join("federated_learning", "data", "data_" + args.experiment)
        self.workers_cluster = []
        file_path = os.path.join(self.local_data_dir, "all_data.json")
        with open(file_path, 'rb') as f:
            data_all = json.load(f)
        for user_data in data_all['user_data'].values():
            self.workers_cluster.append(user_data['cluster_id'])
        print(self.workers_cluster)
        self.pairwise_attach_success = {}


    def print_extraction_performance(self):
        for worker_id, decoded_model_info in self.decoded_models.items():
            self.decoded_models_performance["extraction_accuracy"].append(decoded_model_info["extraction_accuracy"])
            self.decoded_models_performance["train_accuracy"].append(decoded_model_info["train_accuracy"])
            self.decoded_models_performance["test_accuracy"].append(decoded_model_info["test_accuracy"])

        for worker_id, optimal_model_info in self.optimal_models.items():
            self.optimal_models_performance["train_accuracy"].append(optimal_model_info["train_accuracy"])
            self.optimal_models_performance["test_accuracy"].append(optimal_model_info["test_accuracy"])

        Ext_server = []
        Ext_decoded= []
        for worker_id in range(self.num_workers):
            path_dir_test = os.path.join(self.local_data_dir, "train", f"{worker_id}.json")
            with open(path_dir_test, 'rb') as f:
                train_data = json.load(f)
            extra_local_data=(train_data['x'], train_data['y'])
            extract_acc_s = extraction_acc(torch.tensor(self.server_model["model"]),
                                         torch.tensor(self.optimal_models[str(worker_id)]["model"]), self.device,
                                         extra_local_data, self.local_data_dir,
                                         self.args.model)
            extract_acc_d = extraction_acc(torch.tensor(self.decoded_models[str(worker_id)]["model"]),
                                         torch.tensor(self.optimal_models[str(worker_id)]["model"]), self.device,
                                         extra_local_data, self.local_data_dir,
                                         self.args.model)
            Ext_server.append(extract_acc_s)
            Ext_decoded.append(extract_acc_d)
        print("--------------------------------------------")
        print("Extraction Accuracy Train:\n Decoded model:")
        print(Ext_decoded[0:3], sum(Ext_decoded[3:])/7)
        print("--------------------------------------------")
        print("Extraction Accuracy Train:\n Server model:")
        print(Ext_server[0:3], sum(Ext_server[3:])/7)
        print("-------------------------------------------")
        print("Extraction Accuracy:\n Decoded model:")
        print(self.decoded_models_performance["extraction_accuracy"])
        print(sum(self.decoded_models_performance["extraction_accuracy"][3:])/7)
        print("Server model:")
        print(self.server_model["extraction_accuracy"])
        print(sum(self.server_model["extraction_accuracy"][3:])/7)
        print("-------------------------------------------")
        print("Train Accuracy: \n Decoded model:")
        print(self.decoded_models_performance["train_accuracy"])
        print(sum(self.decoded_models_performance["train_accuracy"][3:])/7)
        print("Server model:")
        print(self.server_model["train_accuracy"])
        print(sum(self.server_model["train_accuracy"][3:])/7)
        print("Optimum model:")
        print(self.optimal_models_performance["train_accuracy"])
        print(sum(self.optimal_models_performance["train_accuracy"][3:])/7)
        print("-------------------------------------------")
        print("Test Accuracy: \n Decoded model:")
        print(self.decoded_models_performance["test_accuracy"])
        print(sum(self.decoded_models_performance["test_accuracy"][3:])/7)
        print("Server model:")
        print(self.server_model["test_accuracy"])
        print(sum(self.server_model["test_accuracy"][3:])/7)
        print("Optimum model:")
        print(self.optimal_models_performance["test_accuracy"])
        print(sum(self.optimal_models_performance["test_accuracy"][3:])/7)

    def __attack_chosen_workers(self, workers_to_compare, added_test_local_data = False, even_test_data = True):
        all_x = []
        all_y = []
        true_membership = []
        mem = 0
        file_path = os.path.join(self.local_data_dir, "all_data.json")
        print("---------------------------------------------------------------------------")
        if not added_test_local_data:
            with open(file_path, 'rb') as f:
                data_all = json.load(f)
            for worker_id in workers_to_compare:
                all_x.extend(data_all['user_data'][str(worker_id)]['x'])
                all_y.extend(data_all['user_data'][str(worker_id)]['y'])
                true_membership.extend([mem for i in range(len(data_all['user_data'][str(worker_id)]['y']))])
                mem += 1
        else:
            _, num_classes, num_dimension = get_local_model_structure(self.args.model, self.local_data_dir)
            for worker_id in workers_to_compare:
                train_data, _ = load_extra_local_data(worker_id, self.local_data_dir, dim=num_dimension, num_samples=5000, sigma=0.1, num_classes=num_classes)
                all_x.extend(train_data[0])
                all_y.extend(train_data[1])
                true_membership.extend([mem for i in range(len(train_data[1]))])
                mem += 1

        pre = []
        for worker_id in workers_to_compare:
            optimal_model = self.optimal_models[str(worker_id)]["model"]
            pred_probability, pred_probability_only_after_linear = return_prediction(optimal_model, torch.FloatTensor(all_x), all_y, self.args.model, self.local_data_dir)
            pre.append(pred_probability)

        pre_array = np.array(pre)
        keep_elements = []
        for j in range(pre_array.shape[1]):
            if len(set(pre_array[:, j])) == len(pre_array[:, j]):
                flag = True
                for index, pre_1 in enumerate(pre_array[:,j]):
                    for pre_2 in pre_array[:,j][index+1:]:
                        if abs(pre_1-pre_2) < self.insider_attack_criteria:
                            flag = False
                            break
                if flag:
                    keep_elements.append(j)
        membership = np.argmax(pre, axis=0)
        max_prob = np.max(pre, axis = 0)
        difference = abs(np.array(true_membership) - membership)
        index_chosen = [i for d, i in zip(difference, range(len(difference))) if d == 0 and i in keep_elements and max_prob[i] >= 0.5]
        if len(index_chosen)> 0:
            pre_selected = []
            for pre_individual in pre:
                pre_selected.append([pre_individual[i] for i in index_chosen])
            diff_original = abs(np.array(pre_selected[0]) - np.array(pre_selected[1]))
            diff_original_average = sum(diff_original) / len(diff_original)
            hardness_metric = sum([1 for i in diff_original if i <= 0.3]) / len(diff_original)

            if even_test_data == False:
                print("Chosen distinguished data / distinguished data / all the data/ distinguishable metric/ hardness metric")
                print(f"{len(index_chosen)}/{len(keep_elements)}/{len(pre[0])}/{diff_original_average}/{hardness_metric}")

            all_x_sub = [all_x[i] for i in index_chosen]
            all_y_sub = [all_y[i] for i in index_chosen]
            pre = []
            true_membership_sub = [true_membership[i] for i in index_chosen]
            if even_test_data == True and len(workers_to_compare)==2:
                np.random.seed(0)
                class_1_number = sum(true_membership_sub)
                class_1_index = [ j for i,j in zip(true_membership_sub, range(len(true_membership_sub))) if i==1]
                class_0_number = len(true_membership_sub) - sum(true_membership_sub)
                class_0_index = [ j for i,j in zip(true_membership_sub, range(len(true_membership_sub))) if i==0]

                if class_0_number < class_1_number:
                    class_1_index = np.random.choice(class_1_index, class_0_number, replace=False)
                if class_1_number < class_0_number:
                    class_0_index = np.random.choice(class_0_index, class_1_number, replace=False)

                index_chosen = list(class_0_index)+list(class_1_index)
                true_membership_sub = [true_membership_sub[i] for i in index_chosen]
                all_x_sub = [all_x_sub[i] for i in index_chosen]
                all_y_sub = [all_y_sub[i] for i in index_chosen]
                pre_selected_sub = []
                for pre_individual in pre_selected:
                    pre_selected_sub.append([pre_individual[i] for i in index_chosen])
                diff_original = abs(np.array(pre_selected_sub[0]) - np.array(pre_selected_sub[1]))
                diff_original_average = sum(diff_original)/len(diff_original)
                hardness_metric = sum([1 for i in diff_original if i<=0.3])/len(diff_original)
                print("Even data: class_0_samples/ class_1_samples/ distinguishable metric/ hardness metric")
                print(f"{len(true_membership_sub)-sum(true_membership_sub)}/{sum(true_membership_sub)}/{diff_original_average}/{hardness_metric}")


            elif even_test_data == True and len(workers_to_compare) > 2:
                raise NotImplementedError

            for worker_id in workers_to_compare:
                decoded_model = self.decoded_models[str(worker_id)]["model"]
                pred_probability, pred_linear = return_prediction(decoded_model, torch.FloatTensor(all_x_sub),
                                                                  all_y_sub, self.args.model, self.local_data_dir)
                pre.append(pred_probability)
            prediction_member_ship_sub = np.argmax(pre, axis=0)
            true_positive = sum([1 if actual==0 and predict==0  else 0 for actual, predict in zip(true_membership_sub, prediction_member_ship_sub)])
            false_positive = sum([1 if actual==1 and predict==0  else 0 for actual, predict in zip(true_membership_sub, prediction_member_ship_sub) ])
            false_negative = sum([1 if actual==0 and predict==1  else 0 for actual, predict in zip(true_membership_sub, prediction_member_ship_sub) ])
            false_positive_index = [ind for actual, predict,ind in zip(true_membership_sub, prediction_member_ship_sub, range(len(true_membership_sub))) if actual==1 and predict==0]
            #print([index_chosen[i] for i in false_positive_index])
            #print(len(false_positive_index))
            true_negative = sum([1 if actual == 1 and predict == 1 else 0 for actual, predict in
                                  zip(true_membership_sub, prediction_member_ship_sub)])
            print(f"tp: {true_positive}, fn: {false_negative}, fp: {false_positive}, tn: {true_negative}")

            difference_sub = abs(np.array(true_membership_sub) - np.argmax(pre, axis=0))
            index_chosen_g = [i for d, i in zip(difference_sub, range(len(difference_sub))) if d == 0]
            sucess_rate = len(index_chosen_g) / len(difference_sub)
        else:
            print("Zero data")
            sucess_rate, true_positive, false_negative, false_positive, true_negative, diff_original_average, hardness_metric = 0,0,0,0,0,0,0

        return sucess_rate, len(index_chosen), (true_positive, false_negative, false_positive, true_negative), (diff_original_average, hardness_metric)

    def attack_pair(self, added_test_local_data, even_test_data):
        for i in range(self.num_workers):
            for j in range(i+1, self.num_workers):
                success_rate, test_data_len, confusion_matrix, data_metrics = self.__attack_chosen_workers([i,j],added_test_local_data=added_test_local_data, even_test_data=even_test_data)
                self.pairwise_attach_success.update({str(i)+","+str(j): {"clusters": (self.workers_cluster[i], self.workers_cluster[j]),
                                                             "sucess_rate": success_rate, "num_tested_data": test_data_len,
                                                             "extraction_accuracy": (self.decoded_models_performance["extraction_accuracy"][i],
                                                                                    self.decoded_models_performance["extraction_accuracy"][j]),
                                                            "confusion_matrix":confusion_matrix,
                                                                         "prediction_proba_avg_distance": data_metrics[0],
                                                                         "precentage_of_hard_data_points": data_metrics[1]}})
                print(f'Success pair insider attack with {test_data_len} distinguished data points: {i}/{j} extra:{self.decoded_models_performance["extraction_accuracy"][i]:.2f}/{self.decoded_models_performance["extraction_accuracy"][j]:.2f}'
                      f' cluster:{self.workers_cluster[i]}/{self.workers_cluster[j]}: {success_rate*100:.2f} %')
        """
        sns.distplot(self.pairwise_attach_success)
        plt.xlabel("success rate of pair insider attack")
        plt.ylabel("density")
        plt.title(f"workers {self.num_workers}, local epochs {self.args.num_local_steps}")
        plt.show()
        """
    def attack_chosen(self, chosen_workers, added_test_local_data, even_test_data):
        success_rate, test_data_len, _, _ = self.__attack_chosen_workers(chosen_workers, added_test_local_data=added_test_local_data, even_test_data=even_test_data)
        print(success_rate, test_data_len)

    def attack_all(self, added_test_local_data, even_test_data):
        success_rate, test_data_len, _, _ = self.__attack_chosen_workers([i for i in range(self.num_workers)], added_test_local_data=added_test_local_data, even_test_data=even_test_data)
        print(success_rate, test_data_len)

    def save_results(self):
        self.data.update({"pairwise_insider_attacker": self.pairwise_attach_success})
        with open(self.file_name, 'w') as f:
            json.dump(self.data, f)

    def plot_attacker_performance_paper(self):
        plt.rcParams.update({'font.size': 12})
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if self.args.experiment == "adult":
            X, Y = np.meshgrid([i for i in range(3)], [i for i in range(3, self.num_workers)])
        else:
            X, Y = np.meshgrid([i for i in range(self.num_workers-1)], [i for i in range(1, self.num_workers)])

        ax.plot_surface(X,Y,0*X+0.5, alpha = 0.1)
        if self.args.experiment == "adult":
            ax.set_xticks([i for i in range(3)])
            ax.set_yticks([i for i in range(3, self.num_workers)])
        else:
            ax.set_xticks([i for i in range(self.num_workers - 1)])
            ax.set_yticks([i for i in range(1, self.num_workers)])

        for key, value in self.data["pairwise_insider_attacker"].items():
            i = value["clusters"][0]
            j = value["clusters"][1]
            s = value["sucess_rate"]
            #print(f"{i} vs {j}: sucess={s}, pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points'] * 100:.2f}%")

            if self.args.experiment == "adult":
                if i < 3 and j >= 3:
                    ax.scatter(i, j, s,
                               label=f"pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points'] * 100:.2f}%")
                    confusion_matrix = f"{value['confusion_matrix']}"
                    ax.text(i, j, s, confusion_matrix, fontsize=8)
                    ax.legend(bbox_to_anchor=(0.11, 1.0))
                elif i >= 3:
                    break
            elif self.args.experiment == "purchase_100":
                if s > 0:
                    ax.scatter(int(i), int(j), s,
                               label=f"pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points'] * 100:.2f}%")
                    confusion_matrix = f"{value['confusion_matrix']}"
                    ax.text(int(i), int(j), s, confusion_matrix, fontsize=8)
                    ax.legend()
            else:
                ax.scatter(i,j,s)
                           #label=f"pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points']*100:.2f}%")
                #confusion_matrix= f"{value['confusion_matrix']}"
                #ax.text(i,j,s,confusion_matrix, fontsize=8)
                ax.legend()

        ax.set_xlabel("cluster")
        ax.set_ylabel("cluster")
        ax.set_zlabel("Insider Precision")
        plt.savefig("pairwise_precision_"+self.args.experiment+"_localsteps_"+str(self.args.num_local_steps)+".png", bbox_inches="tight")

        plt.show()

    def plot_attacker_performance(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if self.args.experiment == "adult":
            X, Y = np.meshgrid([i for i in range(3)], [i for i in range(3, self.num_workers)])
        else:
            X, Y = np.meshgrid([i for i in range(self.num_workers-1)], [i for i in range(1, self.num_workers)])

        ax.plot_surface(X,Y,0*X+0.5, alpha = 0.1)
        if self.args.experiment == "adult":
            ax.set_xticks([i for i in range(3)])
            ax.set_yticks([i for i in range(3, self.num_workers)])
        else:
            ax.set_xticks([i for i in range(self.num_workers - 1)])
            ax.set_yticks([i for i in range(1, self.num_workers)])

        for key, value in self.data["pairwise_insider_attacker"].items():
            i = value["clusters"][0]
            j = value["clusters"][1]
            s = value["sucess_rate"]
            print(f"{i} vs {j}: sucess={s}, pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points'] * 100:.2f}%")

            if self.args.experiment == "adult":
                if i < 3 and j >= 3:
                    ax.scatter(i, j, s,
                               label=f"pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points'] * 100:.2f}%")
                    confusion_matrix = f"{value['confusion_matrix']}"
                    ax.text(i, j, s, confusion_matrix, fontsize=8)
                    ax.legend(bbox_to_anchor=(0.11, 1.0))
                elif i >= 3:
                    break
            elif self.args.experiment == "purchase_100":
                if s > 0:
                    ax.scatter(int(i), int(j), s,
                               label=f"pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points'] * 100:.2f}%")
                    confusion_matrix = f"{value['confusion_matrix']}"
                    ax.text(int(i), int(j), s, confusion_matrix, fontsize=8)
                    ax.legend()
            else:
                ax.scatter(i,j,s,label=f"pd={value['prediction_proba_avg_distance']:.2f}, hp={value['precentage_of_hard_data_points']*100:.2f}%")
                confusion_matrix= f"{value['confusion_matrix']}"
                ax.text(i,j,s,confusion_matrix, fontsize=8)
                ax.legend()

        ax.set_xlabel("cluster")
        ax.set_ylabel("cluster")
        ax.set_zlabel("Success rate")

        plt.title(f"{self.num_workers} workers, Local_step: {self.args.num_local_steps}, SP: {self.args.start_point}, ByEpoch: {self.args.fit_by_epoch}\n"
                  f"Extraction accuracy: {[round(i,2) for i in self.decoded_models_performance['extraction_accuracy']]}\n"
                  f"Decode model Train accuracy: {[round(i,2) for i in self.decoded_models_performance['train_accuracy']]}\n"
                  f"Server model Train accuracy: {[round(i,2) for i in self.server_model['train_accuracy']]}\n"
                  f"attack type: {self.ability}, precentage: {self.args.precentage_attack}")

        plt.show()

    def __fn_for_explainer(self, model):
        net, num_classes, num_dimension = get_local_model_structure(self.args.model, self.local_data_dir)
        map_vector_to_net(torch.tensor(model), net, num_classes, num_dimension, self.args.model)
        fn_prediction = lambda x: torch.softmax(net(torch.FloatTensor(x)), dim=1).detach().numpy()
        return fn_prediction

    def interpret_models(self):
        features_names = ['age','educational-num','capital-gain','capital-loss','hours-per-week',
                         'workclass_Federal-gov','workclass_Local-gov',
                         'workclass_Private','workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
                         'workclass_State-gov' ,'workclass_Without-pay', 'education_10th',
                         'education_11th' ,'education_12th', 'education_1st-4th', 'education_5th-6th',
                         'education_7th-8th', 'education_9th', 'education_Assoc-acdm',
                         'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate',
                         'education_HS-grad', 'education_Masters', 'education_Preschool',
                         'education_Prof-school', 'education_Some-college',
                         'marital-status_Divorced', 'marital-status_Married-AF-spouse',
                         'marital-status_Married-civ-spouse',
                         'marital-status_Married-spouse-absent', 'marital-status_Never-married',
                         'marital-status_Separated', 'marital-status_Widowed',
                         'occupation_Adm-clerical', 'occupation_Armed-Forces',
                         'occupation_Craft-repair', 'occupation_Exec-managerial',
                         'occupation_Farming-fishing', 'occupation_Handlers-cleaners',
                         'occupation_Machine-op-inspct', 'occupation_Other-service',
                         'occupation_Priv-house-serv', 'occupation_Prof-specialty',
                         'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support',
                         'occupation_Transport-moving', 'relationship_Husband',
                         'relationship_Not-in-family', 'relationship_Other-relative',
                         'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
                         'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
                         'race_Other', 'race_White', 'gender_Female', 'gender_Male']
        categorical_features = []
        categorical_names = []
        for id, features in enumerate(features_names):
            if len(features.split("_"))>1:
                categorical_features.append(id)
                categorical_names.append(['False', 'True'])

        class_names = ["<50K", ">50K"]
        models = []
        trains = []
        trains_workers = []
        training_labels = []
        training_labels_workers = []
        test_labels = []
        tests = []
        local_file_path = os.path.join(self.local_data_dir, 'train')
        local_file_path_test = os.path.join(self.local_data_dir, 'test')

        fn_prediction_server = self.__fn_for_explainer(self.server_model['model'])

        for worker_id in range(self.num_workers):
            optimal_model = self.optimal_models[str(worker_id)]["model"]
            models.append(optimal_model)
            with open(os.path.join(local_file_path, str(worker_id) + ".json"), 'rb') as f:
                data = json.load(f)
                data_array = np.array(data['x'])
                trains.extend(data_array)
                trains_workers.append(data_array)
                training_labels.extend(data['y'])
                training_labels_workers.append(data['y'])
            with open(os.path.join(local_file_path_test, str(worker_id) + ".json"), 'rb') as f:
                data = json.load(f)
                data_array = np.array(data['x'])
                test_labels.append(data['y'])
                tests.append(data_array)

        trains = np.array(trains)
        training_labels = np.array(training_labels)
        print(trains.shape, training_labels.shape)
        explainer = lime.lime_tabular.LimeTabularExplainer(trains, training_labels=training_labels,
                                                           feature_names=features_names, class_names=class_names,
                                                           categorical_names=categorical_names,
                                                           categorical_features=categorical_features)

        for i in range(self.num_workers):
            if i == 1:
                fn_prediction_optimum = self.__fn_for_explainer(self.optimal_models[str(i)]['model'])
                fn_prediction_decode = self.__fn_for_explainer(self.decoded_models[str(i)]['model'])

                index = 0
                diff_optimum_decode = []
                diff_optimum_server = []
                special_samples = []
                """
                for test_sample, label in zip(trains_workers[i], training_labels_workers[i]):
                #for test_sample, label in zip(tests[i], test_labels[i]):
                    test_sample_reshape = np.array(test_sample).reshape(1, len(test_sample))
                    proba_server = fn_prediction_server(test_sample_reshape)[0][label[0]]
                    proba_optimum = fn_prediction_optimum(test_sample_reshape)[0][label[0]]
                    proba_decode = fn_prediction_decode(test_sample_reshape)[0][label[0]]
                    diff_optimum_decode.append(abs(proba_optimum-proba_decode))
                    diff_optimum_server.append(abs(proba_optimum-proba_server))
                    if abs(proba_optimum-proba_decode)<abs(proba_optimum-proba_server):
                        if abs(proba_optimum-proba_server)-abs(proba_optimum-proba_decode)>0.3:
                            print(index, proba_server, proba_optimum, proba_decode, label)
                            special_samples.append(index)
                    else:
                        if abs(proba_optimum-proba_decode)-abs(proba_optimum-proba_server)>0.1:
                            print("bad")
                            print(index, proba_server, proba_optimum, proba_decode, label)
                            special_samples.append(index)
                    index += 1
                print(special_samples)
                print(sum(diff_optimum_decode)/len(diff_optimum_decode),sum(diff_optimum_server)/len(diff_optimum_server) )
                
                sns.distplot(diff_optimum_decode, label=r"$p_{optimal} - p_{decode}$")
                sns.distplot(diff_optimum_server, label=r"$p_{optimal} - p_{server}$")
                plt.legend()
                plt.title(f"worker {i}")
                plt.show()
                exit(0)
                """
                #sp = submodular_pick.SubmodularPick(explainer, trains_workers[i], fn_prediction)
                #[exp.as_pyplot_figure(label=1) for exp in sp.sp_explanations]
                number = 152
                test_sample = np.array(trains_workers[3][number])
                test_label = training_labels_workers[3][number]
                exp_op = explainer.explain_instance(test_sample, fn_prediction_optimum, num_features=10)
                exp_decode = explainer.explain_instance(test_sample, fn_prediction_decode, num_features=10)
                exp_server = explainer.explain_instance(test_sample, fn_prediction_server, num_features=10)

                test_sample = test_sample.reshape(1, len(trains_workers[3][number]))
                proba_server = fn_prediction_server(test_sample)[0][test_label[0]]
                proba_optimum = fn_prediction_optimum(test_sample)[0][test_label[0]]
                proba_decode = fn_prediction_decode(test_sample)[0][test_label[0]]
                print(proba_server, proba_optimum, proba_decode, test_label)
                """
                if proba_optimum>0.5: prediction_s = ">50K"
                else: prediction_s = "<50K"
                ax = exp_op.as_pyplot_figure()
                plt.title(f"optimum model, label:{test_label[0]}, prediction:{prediction_s}, proba:{proba_optimum*100:.2f} %")
                if proba_decode>0.5: prediction_s = ">50K"
                else: prediction_s = "<50K"
                ax0 = exp_decode.as_pyplot_figure()
                plt.title(f"decoded model, label:{test_label[0]}, prediction:{prediction_s}, proba:{proba_decode*100:.2f} %")
                if proba_server>0.5: prediction_s = ">50K"
                else:
                    prediction_s = "<50K"
                    proba_server = 1-proba_server
                ax1 = exp_server.as_pyplot_figure()
                plt.title(f"server model, label:{test_label[0]}, prediction:{prediction_s}, proba:{proba_server*100:.2f} %")
                """
                exp_decode.save_to_file(f'decoded{i}.html')
                exp_op.save_to_file(f'optimum{i}.html')
                exp_server.save_to_file(f'server{i}.html')
                plt.show()


    def plot_embedding_optimum_model(self):
        data_set = []
        local_file_path = os.path.join(self.local_data_dir, 'train')
        print(local_file_path)
        numbers = []
        data_y = []

        for i in range(10):
            with open(os.path.join(local_file_path, str(i) + ".json"), 'rb') as f:
                data = json.load(f)
                data_array = np.array(data['x'])
                data_y.append(np.array([data['y']]))
                #data_array_with_class = np.c_[data_array, np.array(data['y'])]
                data_array_with_class = data_array
                numbers.append(data_array_with_class.shape[0])
                data_set.extend(data_array_with_class)
        """
        data_set = np.asarray(data_set)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        standard_embedding = umap.UMAP(random_state=42, n_components=3).fit_transform(data_set)
        end = 0
        for worker_id, numbers in zip(range(self.num_workers), numbers):
            start = end
            end += numbers
            x = standard_embedding[start:end, 0]
            y = standard_embedding[start:end, 1]
            z = standard_embedding[start:end, 2]

            ax.scatter(x,y,z, s=1, label=f"worker:{worker_id}")
        plt.legend()
        plt.show()
        """
        models = []
        for worker_id in range(self.num_workers):
            optimal_model = self.optimal_models[str(worker_id)]["model"]
            models.append(optimal_model)
        models = np.array(models)
        print(models.shape)
        """
        distance_vector = []
        distance_vector_random = []
        for i in range(self.num_workers):
            distance_vector.append(np.linalg.norm(models[i] - self.server_model['model']))
            distance_vector_random.append(np.linalg.norm(models[i] - self.init_models[str(i)]))
        print(distance_vector)
        print(distance_vector_random)
        """
        distance_matrix = np.zeros([self.num_workers, self.num_workers])
        for i in range(self.num_workers):
            for j in range(self.num_workers):
                distance_matrix[i,j] = np.linalg.norm(models[i]-models[j]) / np.linalg.norm(models[j])
        plt.rcParams.update({'font.size': 12})
        ax = sns.heatmap(distance_matrix, linewidth=0.5, cmap="YlGnBu", annot=True)
        ax.invert_yaxis()

        plt.xlabel("The index of the client i")
        plt.ylabel("The index of the client j")
        #plt.title('Euclidean distance between optimum models')
        plt.savefig("euc_distance_optimum_models"+self.args.experiment+".png", bbox_inches="tight")
        plt.show()
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        standard_embedding = umap.UMAP(random_state=42, n_components=3).fit_transform(models)
        for worker_id, worker_position in enumerate(standard_embedding):
            x = worker_position[0]
            y = worker_position[1]
            z = worker_position[2]
            ax.scatter(x,y,z, s=100, label=f"worker:{worker_id}")
        plt.legend()
        plt.show()
        """

    def plot_data_point_probability(self, workers_to_compare):

        all_x = []
        all_y = []
        true_membership = []
        mem = 0
        file_path = os.path.join(self.local_data_dir, "all_data.json")
        print("---------------------------------------------------------------------------")
        with open(file_path, 'rb') as f:
            data_all = json.load(f)
        for worker_id in workers_to_compare:
            all_x.extend(data_all['user_data'][str(worker_id)]['x'])
            all_y.extend(data_all['user_data'][str(worker_id)]['y'])
            true_membership.extend([mem for i in range(len(data_all['user_data'][str(worker_id)]['y']))])
            mem += 1
        pre = []
        pre_decode = []
        for worker_id in workers_to_compare:
            optimal_model = self.optimal_models[str(worker_id)]["model"]
            decode_model = self.decoded_models[str(worker_id)]["model"]
            pred_probability, pred_probability_only_after_linear = return_prediction(optimal_model, torch.FloatTensor(all_x), all_y, self.args.model, self.local_data_dir)
            pred_probability_decode, _ = return_prediction(decode_model, torch.FloatTensor(all_x), all_y, self.args.model, self.local_data_dir)
            pre_decode.append(pred_probability_decode)
            pre.append(pred_probability)

        pre_array = np.array(pre)
        keep_elements = []
        for j in range(pre_array.shape[1]):
            if len(set(pre_array[:, j])) == len(pre_array[:, j]):
                flag = True
                for index, pre_1 in enumerate(pre_array[:,j]):
                    for pre_2 in pre_array[:,j][index+1:]:
                        if abs(pre_1-pre_2) < self.insider_attack_criteria:
                            flag = False
                            break
                if flag:
                    keep_elements.append(j)
        membership = np.argmax(pre, axis=0)
        max_prob = np.max(pre, axis = 0)
        difference = abs(np.array(true_membership) - membership)
        index_chosen = [i for d, i in zip(difference, range(len(difference))) if d == 0 and i in keep_elements and max_prob[i] >= 0.5]
        if len(index_chosen)> 0:
            pre_selected = []
            pre_selected_decode = []
            for pre_individual, pre_individual_decode in zip(pre, pre_decode):
                pre_selected.append([pre_individual[i] for i in index_chosen])
                pre_selected_decode.append([pre_individual_decode[i] for i in index_chosen])

            print(len(pre_selected), len(pre_selected_decode), len(pre_selected[0]), len(pre_selected_decode[0]))
            for x_op, y_op, x_de, y_de in zip(pre_selected[0], pre_selected[1], pre_selected_decode[0], pre_selected_decode[1]):
                plt.plot([x_op,x_de], [y_op, y_de], '--', color='grey')

            plt.plot(pre_selected[0], pre_selected[1], '.', color = 'blue', label="optimum model prediction")
            plt.plot(pre_selected_decode[0], pre_selected_decode[1],'x', color = 'red', label="decoded model prediction")
            plt.xlabel(f"Prob(correct label) of worker {workers_to_compare[0]}")
            plt.ylabel(f"Prob(correct label) of worker {workers_to_compare[1]}")
            plt.legend()
            sucess_rate, num_data_points, (true_positive, false_negative, false_positive, true_negative),_ = self.__attack_chosen_workers(workers_to_compare,
                                                                                   added_test_local_data=False,
                                                                                   even_test_data=False)
            plt.plot([0.01*i for i in range(20,100)], [0.01*i for i in range(20,100)],'-.', color='orange')
            plt.title(f"Start from the random model to obtain the decoded model\n Attack success rate {sucess_rate*100:.2f} %, Total number of points: {num_data_points}\n"
                      f"confusion matrix: (w{workers_to_compare[0]}->w{workers_to_compare[0]}:{true_positive}, w{workers_to_compare[0]}->w3:{false_negative}\n"
                      f"w3->w{workers_to_compare[0]}:{false_positive}, w3->w3:{true_negative}")
            plt.show()
