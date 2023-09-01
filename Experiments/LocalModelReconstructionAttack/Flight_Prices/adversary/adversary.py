import os
import json
import torch
import copy
import numpy as np
from utils.args import args_to_string
from adversary.gradient_model.choose_model import get_gradient_model
from adversary.utils.metric import evaluate_loss_gradient_network
from adversary.evals.get_local_model_structure import get_local_model_structure
from adversary.evals.evaluation_decode_ability import evaluate_decoded_model_ability, extraction_acc, evaluate_decoded_model_ability_from_net
from adversary.evals.evaluation_read_extra_local_data import load_extra_local_data, load_optimum_model_to_vector
from adversary.loaders.get_data_loader import get_all_data
from federated_learning.loaders.choose_loaders import get_iterator
from federated_learning.model.choose_model import get_model
from adversary.evals.get_local_model_structure import map_net_to_vector, map_vector_to_gradient_net, freeze_model
from federated_learning.loaders.cifar10 import get_cifar10

class Adversary:
    def __init__(self, args):
        self.ability = args.adversary_ability
        self.num_workers = args.num_workers
        self.gradient_network_type = args.gnetwork_type
        self.num_features = args.gnetwork_features
        self.input_data_dir = os.path.join("logs", self.ability, args_to_string(args))
        self.device = args.device
        self.input_size = 0
        self.num_classes = 0
        self.num_dim = 0 # The local data dimension
        self.sigma = args.sigma
        self.num_epochs = args.gnetwork_num_epochs
        self.trials = args.num_trials_to_decode
        self.loss_func = torch.nn.MSELoss(reduction='sum')
        self.gradient_networks = {i: None for i in range(self.num_workers)}
        self.gradient_networks_loss = {i: None for i in range(self.num_workers)}
        self.gradient_networks_norm = {i: None for i in range(self.num_workers)}
        self.gradient_prediction_mean_std = {i: None for i in range(self.num_workers)}
        self.output_y_norm = {i: None for i in range(self.num_workers)}
        self.output_y_mean_std = {i: None for i in range(self.num_workers)}
        self.fl_lr = args.lr
        self.local_data_dir = os.path.join("federated_learning", "data", "data_" + args.experiment)
        self.experiment = args.experiment
        self.server_model = {'model': None, "test_accuracy": [], "train_accuracy": [], "extraction_accuracy": []}
        self.server_local_model_performance= {i: None for i in range(self.num_workers)}
        self.server_local_models= {i: None for i in range(self.num_workers)}
        self.decoded_models_performance = {i: None for i in range(self.num_workers)}
        self.extra_local_data = {i: None for i in range(self.num_workers)}
        self.optimum_models_performance = {i: None for i in range(self.num_workers)}
        self.args = args
        self.lr = args.adv_lr
        self.start_point = args.start_point
        self.decoded_optimum_epochs = args.decoded_epochs
        self.decoded_optimum_epochs_clients = [self.decoded_optimum_epochs for i in range(self.num_workers)]
        self.init_models = {i: None for i in range(self.num_workers)}
        self.test_loss_best ={i: None for i in range(self.num_workers)}


    def check_if_data_exists(self):
        for i in range(self.num_workers):
            filepath = os.path.join(self.input_data_dir, "inter" + str(i) + ".json")
            if not os.path.isfile(filepath):
                return False
        return True

    def check_if_result_exists(self):
        file_name = self.args.experiment+"_w_"+str(self.args.num_workers)+"_lr_"+str(self.args.lr)+"_bz_"+str(self.args.bz)+"_fit_epoch_"+str(self.args.fit_by_epoch)+"_local_step_"+str(self.args.num_local_steps)\
                    +"_start_point_"+self.args.start_point+"_ability_"+self.ability+"_pre_"+str(self.args.precentage_attack)+"_dp_"+str(self.args.DP)+"_epsilon_"\
                                                                                                                                                      +str(self.args.epsilon)+".json"
        if not os.path.isfile(file_name):
            return False
        else:
            with open(file_name, 'rb') as f:
                data = json.load(f)
            self.gradient_networks = data["gradient_networks"]
            self.gradient_networks_loss = data["gradient_networks_loss"]
            self.optimum_models_performance = data["optimal_model"]
            self.server_model = data["server_model"]
            self.gradient_networks = {int(k): map_vector_to_gradient_net(v,self.gradient_network_type, self.input_size, self.num_features) for k, v in self.gradient_networks.items()}
            self.optimum_models_performance = {int(k): v for k, v in self.optimum_models_performance.items()}
            self.gradient_networks_loss = {int(k): v for k, v in self.gradient_networks_loss.items()}
            return True

    def settle_for_decode_evaluation(self):
        filepath = os.path.join(self.input_data_dir, "inter0.json")
        with open(filepath, 'rb') as f:
            data = json.load(f)
        self.input_size = len(data[0][0])
        if self.ability == "none" or self.ability == "intermediate_attack":
            self.server_model["model"] = data[-1][0]
        elif self.ability == "personalized_attack" or self.ability == "randomized_attack":
            self.server_model["model"] = data[self.args.num_rounds-1][0]
        else:
            raise NotImplementedError
        for c in range(self.num_workers):
            filepath = os.path.join(self.input_data_dir, f"inter{c}.json")
            with open(filepath, 'rb') as f:
                data = json.load(f)
            if self.ability == "randomized_attack":
                self.server_local_models[c] = data[self.args.num_rounds-1][1]
            else:
                self.server_local_models[c] = data[-1][1]



        _, self.num_classes, self.num_dim = get_local_model_structure(self.args.model, self.local_data_dir)

        if self.experiment == "synthetic":
            for worker_id, gradient_model in self.gradient_networks.items():
                extra_train_local_data, extra_test_local_data = load_extra_local_data(worker_id, self.local_data_dir,
                                                                                      dim=self.num_dim, num_samples=100000, sigma=self.sigma, num_classes=self.num_classes)
                self.extra_local_data[worker_id] = {"train_data": None,
                                                    "test_data": extra_test_local_data}
        elif self.experiment == "adult" or self.experiment == "purchase_100" or self.experiment =="synthetic_reg" or self.experiment =="flightPrices":
            for worker_id in range(self.num_workers):
                path_dir_test = os.path.join(self.local_data_dir, "test", f"{worker_id}.json")
                with open(path_dir_test, 'rb') as f:
                    test_data = json.load(f)
                self.extra_local_data[worker_id] = {"train_data": None, "test_data": (test_data['x'], test_data['y'])}
        elif self.experiment == "cifar10":
            cifar10_data, cifar10_targets = get_cifar10()
            for worker_id in range(self.num_workers):
                path_dir_test = os.path.join(self.local_data_dir, "test", f"{worker_id}.json")
                test_data_iterator = get_iterator(self.args.experiment, path_dir_test, self.device, data=cifar10_data, target=cifar10_targets,
                                                  batch_size=self.args.bz, input_type=self.args.model)
                path_dir_train = os.path.join(self.local_data_dir, "train", f"{worker_id}.json")
                train_data_iterator = get_iterator(self.args.experiment, path_dir_train, self.device, data=cifar10_data, target=cifar10_targets,
                                                   batch_size=self.args.bz, input_type=self.args.model)
                self.extra_local_data[worker_id] = {"train_data": train_data_iterator, "test_data": test_data_iterator}
        else:
            raise NotImplementedError

    def clean_useless_states(self):
        dirpath = os.path.join(self.input_data_dir, "inter*.json")
        command = 'rm '+dirpath
        print(command)
        os.system(command)

# add function : exact decoding taking files as input in log



    def train_gradient_network(self):
        seed = 1234
        np.random.seed(seed)
        for worker_id in range(self.num_workers):
            print("Starting getting gradient network of Worker " + str(worker_id) + ":")
            # Get the data for training the gradient network
            x,y = get_all_data(self.input_data_dir, worker_id, self.fl_lr, self.device)
            random_train = []
            for i in range(0,len(x),10):
                random_train.extend(list(np.random.choice([t for t in range(i, i+10)], 9, replace=False)))
            random_train = np.array(random_train)
            random_test = [i for i in range(len(x)) if i not in random_train]
            print(random_test)
            x_train, y_train = x[random_train], y[random_train]
            x_test, y_test = x[random_test], y[random_test]
            best_train_loss = float("inf")

            for t in range(self.trials):
                torch.manual_seed(seed+t)

                # Choose the gradient model type
                print(self.input_size, self.num_features, x_train.size(), y_train.size(), x_test.size(), y_test.size())

                net = get_gradient_model(self.gradient_network_type, self.input_size, self.num_features)
                net.to(self.device)

                # Get the optimizer
                optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0.5)
                the_last_loss = float("inf")
                for epoch in range(self.num_epochs):
                    net.train()
                    prediction = net(x_train)
                    loss = self.loss_func(prediction, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss = evaluate_loss_gradient_network(net,x_train,y_train, self.loss_func)
                    test_loss = evaluate_loss_gradient_network(net,x_test,y_test, self.loss_func)

                    if train_loss < best_train_loss:
                        best_train_loss = train_loss
                        self.test_loss_best[worker_id] = test_loss
                        self.gradient_networks[worker_id] = copy.deepcopy(net.eval())

                    # Early stopping
                    trigger_times=0
                    if test_loss > the_last_loss and self.args.early_stop == True:
                        trigger_times += 1

                        if trigger_times >= self.args.patience or test_loss > 10*train_loss:
                            print(f'Early stopping at epoch {epoch}/ total {self.num_epochs}!')
                            if epoch <= 0.1*self.num_epochs or test_loss > 10*train_loss:
                                self.decoded_optimum_epochs_clients[worker_id] = 1
                            else:
                                self.decoded_optimum_epochs_clients[worker_id] = max(int(epoch/self.num_epochs*self.decoded_optimum_epochs),1)

                            print(self.decoded_optimum_epochs_clients)
                            break
                    else:
                        trigger_times = 0
                    the_last_loss = test_loss

                    if epoch % 1000 == 0:
                        print(f"Epoch: {epoch} | | Train Loss: {train_loss} || Test loss: {test_loss}")

            self.gradient_networks_loss[worker_id] = best_train_loss
            prediction_y = net(x)
            self.gradient_networks_norm[worker_id] = torch.linalg.norm(prediction_y, dim=1).tolist()
            std,mean = torch.std_mean(prediction_y, dim=0)
            print(f"Client {worker_id} prediction: mean {mean}, std {std}")
            self.gradient_prediction_mean_std[worker_id] = (mean.tolist(), std.tolist())
            self.output_y_norm[worker_id] = torch.linalg.norm(y, dim=1).tolist()
            std,mean = torch.std_mean(y, dim=0)
            print(f"Client {worker_id} y: mean {mean}, std {std}")
            self.output_y_mean_std[worker_id] = (mean.tolist(), std.tolist())


    def __decode_local_model(self, worker_id, data_test, data_directory, start_point = "global_model", data_train=None):
        best_cost = float("inf")
        best_decode_model = None
        net, _, _ = get_local_model_structure(self.args.model, self.local_data_dir)

        for t in range(1):
            net_trial = copy.deepcopy(self.gradient_networks[worker_id])
            for param in net_trial.parameters():
                param.requires_grad = False
            net_trial.to(self.device)
            if start_point == "global_model":
                model_initial = torch.reshape(torch.tensor(self.server_local_models[worker_id], device=self.device), (1, self.input_size))
            elif start_point == "random":
                model_initial = torch.zeros((1,self.input_size), requires_grad=True, device=self.device)
                torch.nn.init.xavier_normal_(model_initial)
                self.init_models[worker_id] = model_initial.tolist()
            elif start_point == "zeros":
                model_initial = torch.zeros((1, self.input_size), requires_grad=True, device=self.device)
            else:
                raise NotImplementedError
            print(model_initial.shape)

            model_initial.requires_grad = True

            #optimizer = torch.optim.Adam([model_initial], lr=self.lr)
            optimizer = torch.optim.LBFGS([model_initial], lr=self.lr)
            for i in range(self.decoded_optimum_epochs_clients[int(worker_id)]):
                costs = []
                def closure():
                    optimizer.zero_grad()
                    gradients = net_trial(model_initial)
                    cost = torch.norm(gradients) ** 2 + 0.001*torch.norm(model_initial)**2
                    costs.append(cost.item())
                    cost.backward()
                    # Zero out the convolutaion layer
                    if self.args.model == 'conv':
                        pass
                        #freeze_model(net, model_initial._grad)
                        """
                        update_list = filter(lambda p: p == 0, model_initial._grad[0])
                        update_list_2 = filter(lambda p: p == 0, model_initial.grad[0])
                        print(len(list(update_list)))
                        print(len(list(update_list_2)))
                        exit(0)
                        """
                    return cost
                optimizer.step(closure)
                if costs[-1] < best_cost:
                    best_cost = costs[-1]
                    best_decode_model = model_initial.view(-1)

                if i % 1000 == 0:
                    model_input = model_initial.clone().view(-1)
                    loss, train_acc, test_acc = evaluate_decoded_model_ability(model_input, self.device, worker_id,
                                                                               data_test, data_directory, self.args.model, data_train_iterator=data_train)
                    #print(
                    #    f"Epoch:{i + 1}, Gradient norm: {cost.item():.2f}, Training accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
                    print(f"Epoch:{i+1}, Gradient norm: {costs[-1]:.4f}, Training accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
        print(f"Best: Gradient norm:{best_cost:.4f}")
        return best_decode_model.detach(), best_cost

    def decode_local_models(self):
        for worker_id, gradient_model in self.gradient_networks.items():
            print("\n")
            print("------------------- Server model Performance ---------------------")
            print(f"Worker:{worker_id}, Training accuracy:{self.server_model['train_accuracy'][worker_id]:.2f}, "
                  f"Test accuracy:{ self.server_model['test_accuracy'][worker_id]:.2f}, Extraction Accuracy:{self.server_model['extraction_accuracy'][worker_id]:.2f}")

            print("------------------- Server local model Performance ---------------------")
            print(f"Worker:{worker_id}, Training accuracy:{self.server_local_model_performance[worker_id]['train_accuracy']:.2f}, "
                  f"Test accuracy:{self.server_local_model_performance[worker_id]['test_accuracy']:.2f}, Extraction Accuracy:{self.server_local_model_performance[worker_id]['extraction_accuracy']:.2f}")

            print("------------------- Optimum model Performance ---------------------")
            print(f"Worker:{worker_id}, Training accuracy:{self.optimum_models_performance[worker_id]['train_accuracy']:.2f}, Test accuracy:{self.optimum_models_performance[worker_id]['test_accuracy']:.2f}")

            print("------------------- Decoding procedure starts --------------------")
            if self.ability == "none" or self.ability == "randomized_attack" \
                    or self.ability == "intermediate_attack"\
                    or self.ability == "personalized_attack":
                decoded_model, gradient_of_decode_model = self.__decode_local_model(worker_id, self.extra_local_data[worker_id]["test_data"],
                                                                                    self.local_data_dir, start_point=self.start_point, data_train=self.extra_local_data[worker_id]["train_data"])
            #elif self.ability == "personalized_attack":
            #    x,y = get_all_data(self.input_data_dir, worker_id, self.fl_lr, self.device)
            #    decoded_model = x[-1]
            #    gradient_of_decode_model = 0
            else:
                raise NotImplementedError

            loss_adversary, train_acc_adversary, test_acc_adversary = evaluate_decoded_model_ability(decoded_model,
                                                                                                     self.device,
                                                                                                     worker_id,
                                                                                                     self.extra_local_data[worker_id]["test_data"],
                                                                                                     self.local_data_dir,
                                                                                                     self.args.model, data_train_iterator=self.extra_local_data[worker_id]["train_data"])
            extraction_accuracy = extraction_acc(decoded_model, torch.tensor(self.optimum_models_performance[worker_id]["model"]),
                                                 self.device, self.extra_local_data[worker_id]["test_data"], self.local_data_dir,
                                                 self.args.model)
            self.decoded_models_performance[worker_id] = {"model": decoded_model.tolist(),"train_accuracy": train_acc_adversary, "test_accuracy": test_acc_adversary, "extraction_accuracy": extraction_accuracy, "grad_norm": gradient_of_decode_model}
            print("------------------- Decoded optimum Performance ---------------------")
            if self.ability != "personalized_attack":
                print(f"Worker:{worker_id}, Used Gradient Network Train loss:{self.gradient_networks_loss[worker_id]:.2f}, "
                      f"Test loss: {self.test_loss_best[worker_id]:.2f},"
                      f"Training accuracy: {self.decoded_models_performance[worker_id]['train_accuracy']:.2f}, "
                      f"Test accuracy: {self.decoded_models_performance[worker_id]['test_accuracy']:.2f}, Extraction accuracy: {extraction_accuracy:.2f}")
            else:
                print(f"Worker:{worker_id}, "
                      f"Training accuracy: {self.decoded_models_performance[worker_id]['train_accuracy']:.2f}, "
                      f"Test accuracy: {self.decoded_models_performance[worker_id]['test_accuracy']:.2f}, Extraction accuracy: {self.decoded_models_performance[worker_id]['extraction_accuracy']:.2f}")

    def check_other_benchmark(self):

        # Check the optimum model ability
        for worker_id in range(self.num_workers):
            if self.experiment == "synthetic" and self.args.model == "linear":
                local_optimum = load_optimum_model_to_vector(worker_id, self.local_data_dir)
                local_optimum = torch.tensor(local_optimum).float()
                _, train_acc, test_acc = evaluate_decoded_model_ability(local_optimum, self.device,
                                                                        worker_id,
                                                                        self.extra_local_data[worker_id]["test_data"],
                                                                        self.local_data_dir, self.args.model, data_train_iterator=self.extra_local_data[worker_id]["train_data"])
                self.optimum_models_performance[worker_id] = {"model": local_optimum.tolist(), "train_accuracy": train_acc, "test_accuracy": test_acc}
            elif self.experiment == "adult" or self.experiment == "purchase_100" or self.experiment == "synthetic" or self.experiment == "cifar10" or self.experiment == "synthetic_reg" or self.experiment == "flightPrices" :
                torch.manual_seed(1234)
                path_dir = os.path.join(self.local_data_dir, "train", f"{worker_id}.json")
                path_dir_test = os.path.join(self.local_data_dir, "test", f"{worker_id}.json")
                iter_worker_train = get_iterator(self.args.experiment, path_dir, self.device, self.args.bz, input_type=self.args.model)
                model_worker = get_model(self.args, self.args.model, self.device, iter_worker_train,
                          optimizer_name=self.args.optimizer, lr_scheduler=self.args.lr_scheduler,
                          initial_lr=self.args.lr, epoch_size=self.args.num_rounds)
                for i in range(self.args.num_rounds):
                    if self.args.fit_by_epoch:
                        model_worker.fit_iterator(train_iterator=iter_worker_train, n_epochs=self.args.num_local_steps, verbose=0)
                    else:
                        model_worker.fit_batches(iterator=iter_worker_train, n_steps=self.args.num_local_steps)

                if self.experiment == "synthetic":
                    _, train_acc, test_acc = evaluate_decoded_model_ability_from_net(model_worker.net, self.device,
                                                                        worker_id,
                                                                        self.extra_local_data[worker_id]["test_data"],
                                                                        self.local_data_dir, self.args.model)
                else:
                    iter_worker_train_eval = get_iterator(self.args.experiment, path_dir, self.device, self.args.bz, input_type=self.args.model)
                    _, train_acc = model_worker.evaluate_iterator(iter_worker_train_eval)
                    iter_worker_test = get_iterator(self.args.experiment, path_dir_test, self.device, self.args.bz, input_type=self.args.model)
                    _, test_acc = model_worker.evaluate_iterator(iter_worker_test)
                train_model = map_net_to_vector(model_worker)
                self.optimum_models_performance[worker_id] = {"model": train_model.tolist(),
                                                              "train_accuracy": train_acc, "test_accuracy": test_acc}
                print(f"optimum worker:{worker_id}, train_acc:{train_acc}, test_acc:{test_acc}")
            else:
                raise NotImplementedError

        # Check the server model ability on each of the local data performance
        for worker_id in range(self.num_workers):
            _, train_acc, test_acc = evaluate_decoded_model_ability(torch.tensor(self.server_model["model"]), self.device,
                                                                    worker_id, self.extra_local_data[worker_id]["test_data"],
                                                                    self.local_data_dir, self.args.model, data_train_iterator=self.extra_local_data[worker_id]["train_data"])
            extract_acc = extraction_acc(torch.tensor(self.server_model["model"]), torch.tensor(self.optimum_models_performance[worker_id]["model"]), self.device,
                           self.extra_local_data[worker_id]["test_data"], self.local_data_dir, self.args.model)
            self.server_model["train_accuracy"].append(train_acc)
            self.server_model["test_accuracy"].append(test_acc)
            self.server_model["extraction_accuracy"].append(extract_acc)
            print(f"server worker:{worker_id}, train_acc:{train_acc}, test_acc:{test_acc}, Extrac_acc:{extract_acc}")

            _, train_acc, test_acc = evaluate_decoded_model_ability(torch.tensor(self.server_local_models[worker_id]),
                                                                    self.device, worker_id, self.extra_local_data[worker_id]["test_data"],
                                                                    self.local_data_dir, self.args.model, data_train_iterator=self.extra_local_data[worker_id]["train_data"])
            extract_acc = extraction_acc(torch.tensor(self.server_local_models[worker_id]), torch.tensor(self.optimum_models_performance[worker_id]["model"]), self.device,
                           self.extra_local_data[worker_id]["test_data"], self.local_data_dir, self.args.model)
            self.server_local_model_performance[worker_id] = {"model": self.server_local_models[worker_id],"train_accuracy":train_acc, "test_accuracy":test_acc, "extraction_accuracy":extract_acc}
            print(f"--server local model worker:{worker_id}, train_acc:{train_acc}, test_acc:{test_acc}, Extrac_acc:{extract_acc}")

    def save_results(self, save_type = "lite", results_list = [], local_model_accuracy = [], write_tag=False):
        gradient_networks = {}
        if self.args.adversary_ability != "personalized_attack" and save_type == "heavy":
                for worker_id, gradient_model in self.gradient_networks.items():
                    gradient_networks[worker_id] = map_net_to_vector(gradient_model, type="gradient_network").tolist()
                results = {"decoded_model": self.decoded_models_performance, "server_model": self.server_model,
                           "optimal_model": self.optimum_models_performance, "args": vars(self.args),
                           "gradient_networks":gradient_networks, "gradient_networks_loss":self.gradient_networks_loss,
                           "initial_models": self.init_models}
        else:
            results = {"decoded_model": self.decoded_models_performance, "server_model": self.server_model,
                       "server_local_model": self.server_local_model_performance,
                       "optimal_model": self.optimum_models_performance, "gradient_networks_loss":(self.gradient_networks_loss,self.test_loss_best),
                       "gradient_networks_norm": self.gradient_networks_norm, "output_y_norm": self.output_y_norm,
                       "gradient_prediction_mean_std":self.gradient_prediction_mean_std,
                       "output_y_mean_std":self.output_y_mean_std,
                       "decoded_epochs_clients":self.decoded_optimum_epochs_clients,
                       "local_model_accuracy_FL": local_model_accuracy,
                       "args": vars(self.args)}

        results_list.append(results)

        if write_tag == True:
            file_name = self.args.experiment + "_w_" + str(self.args.num_workers) + "_lr_" + str(
                self.args.lr) + "_bz_" + str(self.args.bz) + "_fit_epoch_" + str(
                self.args.fit_by_epoch) + "_local_step_" + str(self.args.num_local_steps) \
                        + "_start_point_" + self.args.start_point + "_ability_" + self.ability + "_pre_" + str(
                self.args.precentage_attack) + "_dp_" + str(self.args.DP) + "_epsilon_" \
                        + str(self.args.epsilon) + ".json"
            with open(file_name, 'w') as f:
                json.dump(results_list, f)

        return results_list




