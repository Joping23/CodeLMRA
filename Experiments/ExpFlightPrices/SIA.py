#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:28:23 2023


"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
import torch
import torch.nn.functional as F


models = []
losses = []
imgs=[]    


    
d = 8
num_rounds = 1000        
aucs=[]
num_runs = 1
num_workers = 10
data_sample = 5
accuracy_decoded = np.zeros((num_workers))
accuracy_SOTA = np.zeros((num_workers))
aucs = np.zeros((num_runs,num_workers,data_sample))
iteration_LBFGS = 200


num_parameters = 134657             

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
        
net = NeuralLayer(d, 256, 1).to('cpu')
net_SOTA = NeuralLayer(d, 256, 1).to('cpu')



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
    
test_sets = []
y_s = []
losses = np.zeros((10,10))
with open('FP512/gradients.json', 'rb') as f:
    gradients = json.load(f)
    
gradients = np.array(gradients).reshape(10,num_parameters)

for p in range(0,10) :
  
    test_set = pd.read_json('../LocalModelReconstructionAttack/Flight_Prices/federated_learning/data/data_flightPrices/train/'+ str(p)+'.json')
    data_columns =[ 'od', 'round_trip', 'trip_duration', 'month_d', 'week_d',
           'day_of_week_d', 'booking_class', 'tb_departure', 'price']

        
    # comment to take all dataset
    test_set = test_set
    test_sets.append(test_set)
    m = test_set.x.shape[0]
    X = np.ones((m,d))
    y=np.zeros((m,1))
    for i in range(0,m):
        y[i,:] = test_set.y[i]
    y = torch.Tensor(y)
    y_s.append(y)
    
    
for t in range(0,10):       
    for c in range(0,10):
        map_vector_to_net(torch.Tensor(gradients[c]),net_SOTA,num_classes=1,num_dimension=8,model="neural")
        criterion = torch.nn.MSELoss()
        output = net_SOTA(torch.Tensor(test_sets[t].x))
        losse = criterion(output, y_s[t])
        losses[c,t] = losse
        source = np.argmin(np.array(losses),axis=0)

    
        
print("True sources : [0 1 2 3 4 5 6 7 8 9]")
print("Predicted Sources using  SOTA")

print(source)


   
   
results =  pd.read_json('flightPrices_w_10_lr_5e-06_bz_512_fit_epoch_True_local_step_1_start_point_global_model_ability_none_pre_0.1_dp_False_epsilon_1.0.json')

losses_decoded = np.zeros((10,10)) 
models = []   
for i in range(0,10)  :
   models.append(results.decoded_model[0][str(i)]['model'])

for i in range(0,10)  :
    for j in range(0,10) :
            
        # Model-based SIA
        net = NeuralLayer(8, 256, 1).to('cpu')
        map_vector_to_net(torch.Tensor(models[i]),net,num_classes=1,num_dimension=8,model="neuralReg")
        criterion = torch.nn.MSELoss()
        output = net(torch.Tensor(test_sets[j].x))
        losse = criterion(output, y_s[j])
        losses_decoded[i,j] = losse
          
print("True sources : [0 1 2 3 4 5 6 7 8 9]")
print("Predicted Sources using decoded model")

print(np.argmin(losses_decoded,axis=0))




losses_last = np.zeros((10,10)) 
models = []   
for i in range(0,10)  :
   models.append(results.server_local_model[0][str(i)]['model'])

for i in range(0,10)  :
    for j in range(0,10) :
            
        # Model-based SIA
        net_last = NeuralLayer(8, 256, 1).to('cpu')
        map_vector_to_net(torch.Tensor(models[i]),net_last,num_classes=1,num_dimension=8,model="neuralReg")
        criterion = torch.nn.CrossEntropyLoss()
        output = net_last(torch.Tensor(test_sets[j].x))
        losse = criterion(output, y_s[j])
        losses_last[i,j] = losse
        
print("True sources : [0 1 2 3 4 5 6 7 8 9]")
print("Predicted Sources using last returned model")

print(np.argmin(losses_last,axis=0))


   

