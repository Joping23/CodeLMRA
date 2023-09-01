# -*- coding: utf-8 -*-

#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json

models = []

for worker_id in range(0,10)    :

    results =  pd.read_json('adult_w_10_lr_0.05_bz_512_fit_epoch_False_local_step_10_start_point_global_model_ability_none_pre_0.1_dp_False_epsilon_1.0.json')
    models.append(results.decoded_model[0][str(worker_id)]['model'])
    
d = 62
num_rounds = 200
aucs=[]
num_runs = 1
num_workers = 10
data_sample = 5
accuracy_decoded = np.zeros((num_workers))
accuracy_SOTA = np.zeros((num_workers))
aucs = np.zeros((num_runs,num_workers,data_sample))
iteration_LBFGS = 200
sensitive_id = 43  # 39 for gender #54 for marital status


temperature = 1 


#interval = [[5,5],[10,10],[25,25],[50,50],[500,500]]
interval = [[500,500]]
class NeuralLayer(nn.Module):
    def __init__(self, input_dimension, intermediate_dimension, num_classes):
        super(NeuralLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.intermediate_dimension = intermediate_dimension
        self.fc1 = nn.Linear(input_dimension, intermediate_dimension)
        self.fc2 = nn.Linear(intermediate_dimension, intermediate_dimension)
        self.fc3 = nn.Linear(intermediate_dimension, intermediate_dimension)

        self.fc4 =  nn.Linear(intermediate_dimension, num_classes)

        self.relu = nn.ReLU()
    def forward(self, x):
        first_layer = self.relu(self.fc1(x))
        second_layer = self.relu(self.fc2(first_layer))
        third_layer = self.relu(self.fc3(second_layer))
        output = self.fc4(third_layer)
        return output

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        
def _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
    """

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
    :param reconstructed_data:
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
        features and by continuous features separately.
    :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
        If the flag 'detailed' is set to True the reconstruction errors of the categorical and the continuous features
        are returned separately.
    """
    cat_score = 0
    cont_score = 0
    num_cats = 0
    num_conts = 0

    for true_feature, reconstructed_feature, tol in zip(true_data, reconstructed_data, tolerance_map):
        if tol == 'cat':
            cat_score += 0 if str(true_feature) == str(reconstructed_feature) else 1
            num_cats += 1
        elif not isinstance(tol, str):
            cont_score += 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
            num_conts += 1
        else:
            raise TypeError('The tolerance map has to either contain numerical values to define tolerance intervals or '
                            'the string >cat< to mark the position of a categorical feature.')
    if detailed:
        if num_cats < 1:
            num_cats = 1
        if num_conts < 1:
            num_conts = 1
        return (cat_score + cont_score)/(num_cats + num_conts), cat_score/num_cats, cont_score/num_conts
    else:
        return (cat_score + cont_score)/(num_cats + num_conts)
        
net_SOTA = NeuralLayer(d, 256, 2).to('cpu')

def map_vector_to_net(adversary_local_model, net, num_classes, num_dimension, model):
    if model == "linear" or model == 'linearRegression':
        iter = 0
        for param in net.parameters():
            if iter == 0:
                to_fix = adversary_local_model[:-num_classes]
                print(to_fix)
                param.data = to_fix.reshape(num_classes, num_dimension)
            else:
                param.data = adversary_local_model[-num_classes:]
            iter += 1
    elif model == "neural" or model == "mlp" or model == "conv" or model =="neuralReg":
        start = 0
        for param in net.parameters():
            size_of_param = np.prod(list(param.data.shape))
            end = start + size_of_param
            to_fix = adversary_local_model[start:end]
            param.data = to_fix.reshape(param.data.shape)
            start = end
    else:
        raise NotImplementedError
    

for j in range(0,10) :
    worker_id = j
    # load gradients from latest experiment 
    
    # with open('../LocalModelReconstructionAttack/Adult&Synthetic/logs/none/experiment_adult_bz_128_lr_0.01_lr_scheduler_constant_optimizer_sgd_fit_by_epoch_False_num_local_steps_1_precentage_attack_0.1_DP_False_epsilon_1.0/inter' + str(worker_id)+'.json', 'rb') as f:
    #     data_worker = json.load(f)
    gradients = np.load('gradients.npy')
    test_set = pd.read_json('../LocalModelReconstructionAttack/Adult&Synthetic/federated_learning/data/data_adult/train/'+ str(worker_id)+'.json')

    # comment to choose all dataset
    test_set = test_set.head(200)
    m = test_set.x.shape[0]
    X = np.zeros((m,d))
    for i in range(0,m):
        X[i,:] = test_set.x[i]
    y=np.zeros((m,1))
    for i in range(0,m):
        y[i,:] = test_set.y[i]
    
    y = torch.Tensor(y)

    X_p = torch.cat([torch.Tensor(X[:,0:sensitive_id]),torch.Tensor(X[:,sensitive_id+1:d])],dim=1)
    torch.manual_seed(0)
    logits = torch.randn((m,2)).requires_grad_(True)
    
    #print(logits[:,1].sum())
    optimizer = torch.optim.SGD([logits],lr=0.001)
    for iter in range(0,500):
        optimizer.zero_grad()
        grad_diff = 0 
        p=F.gumbel_softmax(logits, tau=1, hard=False)
        X_s = p[:,1].reshape(m,1)          
        
        # X_s is a random variable follows an unnormalized probability vector logits 
        for t in range(0,10):           
            
              map_vector_to_net(torch.Tensor(gradients[(5*worker_id)+t:(5*worker_id) + t+1,:])[0],net_SOTA,num_classes=2,num_dimension=d,model="neural")
              # uncomment to use latest exchanged models
              # map_vector_to_net(torch.Tensor(data_worker[t][1]),net_SOTA,num_classes=2,num_dimension=d,model="neural")

              criterion = torch.nn.CrossEntropyLoss()
              output = net_SOTA(torch.Tensor(X))
              # print(torch.argmax(output))
              y_pred = criterion(output, F.one_hot(y.to(torch.int64),2)[:,0,:].float())
              #print(y_pred)
              original_gradients = torch.autograd.grad(y_pred, net_SOTA.parameters())
              # original_gradients = list((_.detach().clone() for _ in original_gradient))
              X_dum = torch.cat([X_p,X_s],dim=1)
              X_dum[[sensitive_id,d]] = X_dum[[d,sensitive_id]] 

              pred = net_SOTA(X_dum)
              #print(pred)

              dummy_loss = criterion(pred,F.one_hot(y.to(torch.int64),2)[:,0,:].float())
              # print(dummy_loss)
              dummy_gradient = torch.autograd.grad(dummy_loss, net_SOTA.parameters(), create_graph=True)
              grad_diff = 0 
          #   for gx, gy in zip(dummy_gradient[1], original_gradients[1]):  use for MSE Loss
          #     grad_diff += ((gx - gy) ** 2).sum()
            # grad_diff.backward(retain_graph=True)
                
              grad_diff +=torch.sum(torch.nn.functional.cosine_similarity(original_gradients[0].reshape(256,d),dummy_gradient[0].reshape(256,d)))
              grad_diff = -grad_diff
              grad_diff.backward(retain_graph=True)
              optimizer.step()
          #if iter%100==0: print(f"Worker {worker_id} Iteration {iter} Loss {grad_diff.item()}")
    X_s = logits[:,1]
    X_pred = 1*(X_s.detach().numpy() > 0.5)
    accuracy_SOTA[worker_id] = (sum(X_pred==X[:,sensitive_id])/len(X_pred))
    print(f"Accuracy of attacking worker {worker_id} using SOTA is {accuracy_SOTA[worker_id]}")
        #aucs[r,worker_id] = (metrics.roc_auc_score(X[:,1],X_s.detach().numpy()))
   
    # Model-based AIA
    net = NeuralLayer(d, 256, 2).to('cpu')
    map_vector_to_net(torch.Tensor(models[j]),net,num_classes=1,num_dimension=d,model="neuralReg")
    accuracies = np.zeros(m)
    candid_zero = X.copy()
    candid_zero[:,sensitive_id] = np.zeros(m)
    candid_one = X.copy()
    candid_one[:,sensitive_id] = np.ones(m)
    
    for i in range(0,m):
        if criterion(net(torch.Tensor(candid_one)[i]),F.one_hot(y[i].to(torch.int64),2)[0].float())< criterion(net(torch.Tensor(candid_zero)[i]),F.one_hot(y[i].to(torch.int64),2)[0].float() ):
            accuracies[i] = 1
            
    acc_decoded = sum(X[:,sensitive_id]==accuracies)/m      
    print(f"Accuracy of attacking worker {worker_id} using decoded local model {worker_id} is : " + str(sum(X[:,sensitive_id]==accuracies)/m      ))
        
    
