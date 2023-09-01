
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy import spatial

models_decoded = []
models_optimal= []
models_last = []
global_model =[]
num_runs = 1
num_workers = 10
data_sample = 5
iteration_LBFGS = 200  
d = 8
num_rounds = 200




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
        
net_decoded = NeuralLayer(8, 256, 1).to('cpu')
net_optimal = NeuralLayer(8, 256, 1).to('cpu')
net_global = NeuralLayer(8, 256, 1).to('cpu')
net_last = NeuralLayer(8, 256, 1).to('cpu')


for worker_id in range(0,10)    :
    results =  pd.read_json('flightPrices_w_10_lr_5e-06_bz_512_fit_epoch_True_local_step_1_start_point_global_model_ability_none_pre_0.1_dp_False_epsilon_1.0.json')
    models_decoded.append(results.decoded_model[0][str(worker_id)]['model'])
    models_optimal.append(results.optimal_model[0][str(worker_id)]['model'])
    global_model.append(results.server_model[0]['model'])
    models_last.append(results.server_local_model[0][str(worker_id)]['model'])







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
  

  
    test_set = pd.read_json('../LocalModelReconstructionAttack/Flight_Prices/federated_learning/data/data_flightPrices/train/'+ str(worker_id)+'.json')
    data_columns =[ 'od', 'round_trip', 'trip_duration', 'month_d', 'week_d',
           'day_of_week_d', 'booking_class', 'tb_departure', 'price']

    # comment to take all dataset
    test_set = test_set.head(1000)
    m = test_set.x.shape[0]
    X = np.ones((m,d))
    y=np.zeros((m,1))
    for i in range(0,m):
        y[i,:] = test_set.y[i]
    for i in range(0,m):
        X[i,:] = test_set.x[i]
    
    y=np.zeros((m,1))
    for i in range(0,m):
        y[i,:] = test_set.y[i]

    
    
    # Model-based AIA
    map_vector_to_net(torch.Tensor(models_decoded[j]),net_decoded,num_classes=1,num_dimension=8,model="neuralReg")
    map_vector_to_net(torch.Tensor(models_optimal[j]),net_optimal,num_classes=1,num_dimension=8,model="neuralReg")
    map_vector_to_net(torch.Tensor(global_model[j]),net_global,num_classes=1,num_dimension=8,model="neuralReg")
    map_vector_to_net(torch.Tensor(models_last[j]),net_last,num_classes=1,num_dimension=8,model="neuralReg")
    
    # Define the prediction function
    def predict_fn_decoded(x):
        return net_decoded(torch.Tensor(x)).detach().numpy()
    
    def predict_fn_optimal(x):
        return net_optimal(torch.Tensor(x)).detach().numpy()
    
    def predict_fn_global(x):
        return net_global(torch.Tensor(x)).detach().numpy()
    def predict_fn_last(x):
        return net_last(torch.Tensor(x)).detach().numpy()
    
    # Initialize the LIME explainer and generate explanations
    import lime as lime
    exp_decoded=[]
    exp_last=[]
    exp_global =[]
    exp_optimal=[]

    batch = 1000
    explainer = lime.lime_tabular.LimeTabularExplainer(X, mode='regression')
    for i in range(0,batch) :
        explanation_decoded = explainer.explain_instance(X[i], predict_fn_decoded)
        exp_decoded.append(np.array(explanation_decoded.local_exp[0])[:,1])
        
        explanation_optimal = explainer.explain_instance(X[i], predict_fn_optimal)
        exp_optimal.append(np.array(explanation_optimal.local_exp[0])[:,1])
        
        explanation_global = explainer.explain_instance(X[i], predict_fn_global)
        exp_global.append(np.array(explanation_global.local_exp[0])[:,1])
        
        explanation_last = explainer.explain_instance(X[i], predict_fn_last)
        exp_last.append(np.array(explanation_last.local_exp[0])[:,1])
        
    weights_decoded = np.array(exp_decoded).reshape(batch*8)
    weights_optimal = np.array(exp_optimal).reshape(batch*8)
    weights_global = np.array(exp_global).reshape(batch*8)
    weights_last = np.array(exp_last).reshape(batch*8)

 

    print(f"Distance between decoded local model of worker {worker_id} and its optimal model is {np.abs(1-spatial.distance.cosine(weights_decoded, weights_optimal))}")
    print(f"Distance between last model of worker {worker_id} and its optimal model is {np.abs(1-spatial.distance.cosine(weights_last, weights_optimal))}")
    print(f"Distance between global model of worker {worker_id} and its optimal model is {np.abs(1-spatial.distance.cosine(weights_global, weights_optimal))}")
    
    
