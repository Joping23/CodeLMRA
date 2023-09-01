import os
import torch
from federated_learning.loaders.cifar10 import get_cifar10
from federated_learning.loaders.choose_loaders import get_iterator
from federated_learning.model.choose_model import get_model
from utils.args import args_to_string
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json


class FederatedLearningFramework:
    def __init__(self, args, seed=1234):
        self.args = args
        self.num_workers = args.num_workers
        self.num_rounds = args.num_rounds
        self.fit_by_epoch = args.fit_by_epoch
        self.num_local_steps = args.num_local_steps
        self.lr = args.lr
        self.bz = args.bz
        self.optimizer = args.optimizer
        self.experiment = args.experiment
        self.lr_scheduler = args.lr_scheduler
        self.device = args.device
        self.round_idx = 0
        self.intermediate_state = {k: [] for k in range(self.num_workers)}
        self.log_freq = args.log_freq
        self.logger_dir = os.path.join("logs", args.adversary_ability)
        self.adversary_ability = args.adversary_ability
        self.precentage_attack = args.precentage_attack
        self.local_model_accuracy = {k: [] for k in range(self.num_workers)}
        if self.adversary_ability == "personalized_attack" or self.adversary_ability == "randomized_attack":
            self.old_rounds = self.num_rounds
            self.num_rounds = int((1+self.precentage_attack)*args.num_rounds)
            #self.old_rounds = 0
            #self.num_rounds = args.num_rounds
            print(f"Number of rounds in total: {self.num_rounds}")
        elif self.adversary_ability == "intermediate_attack":
            attack_times = int(self.num_rounds*self.precentage_attack)
            interval = int(self.num_rounds/attack_times)
            self.attack_round_idx = [i*interval for i in range(attack_times)]

        os.makedirs(self.logger_dir, exist_ok=True)
        self.logger_path = os.path.join(self.logger_dir, args_to_string(args))
        self.logger = SummaryWriter(self.logger_path)
        #check the federated learning seed

        # Read Data loaders for all the train/test datasets
        self.train_dir = os.path.join("federated_learning", "data", "data_" + args.experiment, "train")
        self.test_dir = os.path.join("federated_learning", "data", "data_" + args.experiment, "test")
        self.train_file_path = os.path.join(self.train_dir, "train.json")
        self.test_file_path = os.path.join(self.test_dir, "test.json")
        #These two iterators are for evaluation
        if args.experiment != "cifar10":
            self.train_iterator = get_iterator(args.experiment, self.train_file_path, self.device, self.bz)
            self.test_iterator = get_iterator(args.experiment, self.test_file_path, self.device, self.bz)
        else:
            cifar10_data, cifar10_targets = get_cifar10()
            self.train_iterator = get_iterator(args.experiment, self.train_file_path, self.device,
                                               data=cifar10_data, target=cifar10_targets, batch_size=self.bz, input_type=self.args.model)
            self.test_iterator = get_iterator(args.experiment, self.test_file_path, self.device,
                                              data=cifar10_data, target=cifar10_targets, batch_size= self.bz, input_type=self.args.model)
            print(len(self.train_iterator.dataset), len(self.test_iterator.dataset))
        # Read Data loaders for each of the worker
        self.workers_iterators = []
        self.local_function_weights = np.zeros(self.num_workers)
        train_data_size = 0
        for worker_id in range(self.num_workers):
            data_path = os.path.join(self.train_dir, str(worker_id) + ".json")
            if args.experiment != "cifar10":
                self.workers_iterators.append(get_iterator(args.experiment, data_path, self.device, self.bz, args.DP))
            else:
                self.workers_iterators.append(get_iterator(args.experiment, data_path, self.device,
                                                           data=cifar10_data, target=cifar10_targets, batch_size=self.bz, input_type=self.args.model))
            print(worker_id, len(self.workers_iterators[-1].dataset))
            train_data_size += len(self.workers_iterators[-1])
            self.local_function_weights[worker_id] = len(self.workers_iterators[-1].dataset)

        self.epoch_size = int(train_data_size / self.num_workers)
        self.local_function_weights = self.local_function_weights / self.local_function_weights.sum()

        # Build workers states
        if args.use_weighted_average:
            self.workers_models = [get_model(args, args.model, self.device, self.workers_iterators[w_i],
                                             optimizer_name=self.optimizer, lr_scheduler=self.lr_scheduler,
                                             initial_lr=self.lr, epoch_size=self.epoch_size,
                                             coeff=self.local_function_weights[w_i], seed=seed)
                                   for w_i in range(self.num_workers)]
        else:
            self.workers_models = [get_model(args, args.model, self.device, self.workers_iterators[w_i],
                                             optimizer_name=self.optimizer, lr_scheduler=self.lr_scheduler,
                                             initial_lr=self.lr, epoch_size=self.epoch_size, seed=seed)
                                   for w_i in range(self.num_workers)]

        # average model of all workers
        self.global_model = get_model(args, args.model,
                                      self.device,
                                      self.train_iterator,
                                      epoch_size=self.epoch_size, seed=seed)

    def log_iter_model(self, worker_id, model, state_data="input"):
        if state_data == "output":
            _, train_acc = model.evaluate_iterator(self.workers_iterators[worker_id])
            self.local_model_accuracy[worker_id].append(train_acc)
        if state_data == "input":
            self.intermediate_state[worker_id].append([])
        iter_num = 0
        for param in model.net.parameters():
            if iter_num == 0:
                train_model = param.data.clone().view(-1)
            else:
                train_model = torch.cat((train_model, param.data.clone().view(-1)))
            iter_num += 1
        self.intermediate_state[worker_id][-1].append(train_model.tolist())

    def write_logs(self):
        """
        write train/test loss, train/tet accuracy for average model and local models
         and intra-workers parameters variance (consensus) adn save average model
        """
        train_loss, train_acc = self.global_model.evaluate_iterator(self.train_iterator)
        test_loss, test_acc = self.global_model.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.round_idx)
        self.logger.add_scalar("Train/Acc", train_acc, self.round_idx)
        self.logger.add_scalar("Test/Loss", test_loss, self.round_idx)
        self.logger.add_scalar("Test/Acc", test_acc, self.round_idx)

        # write parameter variance
        average_parameter = self.global_model.get_param_tensor()

        param_tensors_by_workers = torch.zeros((average_parameter.shape[0], self.num_workers))

        for ii, model in enumerate(self.workers_models):
            param_tensors_by_workers[:, ii] = model.get_param_tensor() - average_parameter

        consensus = (param_tensors_by_workers ** 2).mean()
        self.logger.add_scalar("Consensus", consensus, self.round_idx)

        print(f'\t Round: {self.round_idx} |Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

    def write_intermediate_state(self):
        for worker_id in range(self.num_workers):
            with open(self.logger_path + '/inter' + str(worker_id) + '.json', 'w') as f:
                json.dump(self.intermediate_state[worker_id], f)
                print("Finish writing to " + self.logger_path)

    def launch(self):
        """
        All the local models are averaged, and the average model is re-assigned to each work
        """
        for round_idx in range(self.num_rounds):
            for worker_id, model in enumerate(self.workers_models):
                model.net.to(self.device)

                self.log_iter_model(worker_id, model, state_data="input")

                if self.fit_by_epoch:
                    model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                       n_epochs=self.num_local_steps, verbose=0)
                else:
                    model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.num_local_steps)

                self.log_iter_model(worker_id, model, state_data="output")

            for param_idx, param in enumerate(self.global_model.net.parameters()):
                param.data.fill_(0.)
                for worker_model in self.workers_models:
                    param.data += (1 / self.num_workers) * list(worker_model.net.parameters())[param_idx].data.clone()

            if self.adversary_ability == "personalized_attack" and round_idx >= self.old_rounds-1:
                pass
            elif self.adversary_ability == "randomized_attack" and round_idx >= self.old_rounds-1:
                for ii, model in enumerate(self.workers_models):
                    model.net.reset_parameters()
            elif self.adversary_ability == "intermediate_attack" and round_idx in self.attack_round_idx:
                for ii, model in enumerate(self.workers_models):
                    model.net.reset_parameters()
            else:
                for ii, model in enumerate(self.workers_models):
                    for param_idx, param in enumerate(model.net.parameters()):
                        param.data = list(self.global_model.net.parameters())[param_idx].data.clone()

            self.round_idx += 1

            if (self.round_idx - 1) % self.log_freq == 0:
                self.write_logs()

        self.write_intermediate_state()
        return self.local_model_accuracy