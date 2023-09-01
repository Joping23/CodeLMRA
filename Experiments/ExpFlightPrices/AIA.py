# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import json 
import pandas as pd
import numpy as np

models = []
losses = []
imgs=[]

    

for worker_id in range(0,10)    :
    results =  pd.read_json('flightPrices_w_10_lr_5e-06_bz_512_fit_epoch_True_local_step_1_start_point_global_model_ability_none_pre_0.1_dp_False_epsilon_1.0.json')
    models.append(results.decoded_model[0][str(worker_id)]['model'])
    
d = 8
num_rounds = 1000
num_parameters = 134657             
num_runs = 1
num_workers = 10
data_sample = 5
accuracy_decoded = np.zeros((num_workers))
accuracy_SOTA = np.zeros((num_workers))
aucs = np.zeros((num_runs,num_workers,data_sample))
iteration_LBFGS = 200



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
        
net_SOTA = NeuralLayer(8, 256, 1).to('cpu')



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
data_workers=[]    

for j in range(0,10) :
    worker_id = j
        
    
    
    with open('FP512/gradients.json', 'rb') as f:
            gradients = json.load(f)
        
    gradients = np.array(gradients).reshape(10,num_parameters)
    test_set = pd.read_json('../LocalModelReconstructionAttack/Flight_Prices/federated_learning/data/data_flightPrices/train/'+ str(worker_id)+'.json')
    data_columns =[ 'od', 'round_trip', 'trip_duration', 'month_d', 'week_d',
           'day_of_week_d', 'booking_class', 'tb_departure', 'price']

    # comment to take all dataset
    test_set = test_set
    m = test_set.x.shape[0]
    X = np.ones((m,d))
    y=np.zeros((m,1))
    for i in range(0,m):
        y[i,:] = test_set.y[i]

    for i in range(0,m):
        X[i,:] = test_set.x[i]
    y = torch.Tensor(y)
    X_p = torch.Tensor(X[:,0:7])
    torch.manual_seed(0)
    logits = torch.randn((m,2)).requires_grad_(True)
    
    #print(logits[:,1].sum())
    optimizer = torch.optim.SGD([logits],lr=0.01)
    
    history = []
    for iter in range(0,500):
        optimizer.zero_grad()
        grad_diff = 0 
        p=F.gumbel_softmax(logits, tau=1, hard=False)
        X_s = p[:,1].reshape(m,1)          
        
        # X_s is a random variable follows an unnormalized probability vector logits 
        
        for t in range(0,10):                   
             map_vector_to_net(torch.Tensor(gradients[t]),net_SOTA,num_classes=1,num_dimension=d,model="neural")

             criterion = torch.nn.MSELoss()
             output = net_SOTA(torch.Tensor(X))
             y_pred = criterion(output, y)
             original_gradients = torch.autograd.grad(y_pred, net_SOTA.parameters())
             X_dum = torch.cat([X_p,X_s],dim=1)
             X_dum[:,[1,7]] = X_dum[:,[7,1]]
             pred = net_SOTA(X_dum)
             dummy_loss = criterion(pred,y)
             dummy_gradient = torch.autograd.grad(dummy_loss, net_SOTA.parameters(), create_graph=True)
             grad_diff = 0 
             grad_diff +=torch.sum(torch.nn.functional.cosine_similarity(original_gradients[0].reshape(256,d),dummy_gradient[0].reshape(256,d)))
             grad_diff = -grad_diff
             grad_diff.backward(retain_graph=True)
             optimizer.step()
             
             
    #print(logits[:,1].sum())
    X_s = logits[:,1]
    X_pred = 1*(X_s.detach().numpy() > 0.5)
    accuracy_SOTA[worker_id] = (sum(X_pred==X[:,1])/len(X_pred))
    print(f"Accuracy of attacking worker {worker_id} using SOTA  is {accuracy_SOTA[worker_id]}")
   
    
    
    
    # Model-based AIA
    net = NeuralLayer(8, 256, 1).to('cpu')
    map_vector_to_net(torch.Tensor(models[j]),net,num_classes=1,num_dimension=8,model="neuralReg")
    accuracies = np.zeros(m)
    candidate0 = torch.cat([X_p,torch.zeros(m,1)],dim=1)
    candidate0[:,[1,7]] = candidate0[:,[7,1]]
        
    
    
    candidate1 = torch.cat([X_p,torch.ones(m,1)],dim=1)
    candidate1[:,[1,7]] = candidate1[:,[7,1]]
    
    for i in range(0,m):
        if (y[i] - net(candidate1)[i])**2 < (y[i] - net(candidate0)[i])**2 :
            accuracies[i] = 1
            
    acc_decoded = sum(X[:,1]==accuracies)/m      
    print(f"Accuracy of attacking worker {worker_id} using decoded local model is {acc_decoded}")
        
    
