

import torch

import pandas as pd
import torch.nn.functional as F

from torchvision import  transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity     
import json
from PIL import Image
from model import Model


class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 4)
        self.conv3 = nn.Conv2d(16, 32, 4)
        self.conv4 = nn.Conv2d(32, 64, 4)
        self.conv5 = nn.Conv2d(64, 128, 4)

        self.fc1 = nn.Linear(3840, 64)
        self.fc3 = nn.Linear(64, 41)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
#        x=self.pool(x)

        x = F.relu(self.conv3(x))
        x=self.pool(x)
        x= F.relu(self.conv4(x))
        x=self.pool(x)

        x= F.relu(self.conv5(x))

        x=self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        
        self.fc1.reset_parameters()
        self.fc3.reset_parameters()



net = ConvLayer().to('cpu')
net_sub = ConvLayer().to('cpu') 

data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            
            transforms.Normalize(mean=[0.485,],
                                 std=[0.229,])
        ])



def map_vector_to_net(adversary_local_model, net, num_classes, num_dimension, model):
    if model == "linear" or model == 'linearRegression':
        iter = 0
        for param in net.parameters():
            if iter == 0:
                to_fix = adversary_local_model[:-num_classes]
                # print(to_fix)
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
            #print(param.data.shape)
            param.data = to_fix.reshape(param.data.shape)
            #print(f"{param.data.size()},{adversary_local_model[start:end].size()}")
            start = end
    else:
        raise NotImplementedError

for worker_id in range(0,1) : 
    with open('inter'+str(worker_id)+'.json', 'rb') as f:
            data_worker = json.load(f)


    with open('train/train.json', 'rb') as f:

            reel_imgs = json.load(f)
    begin=0        
    bz = 128
    #label = reel_imgs["y"][begin:begin+bz]
    imgs = torch.FloatTensor(np.array(reel_imgs["x"][begin:begin+bz]).reshape(bz,80,70))
   # reel_img = torch.Tensor(np.array(Image.open(str(label)+".jpg")))
    im_to_recover = torch.randn(bz,80,70).requires_grad_(True)
    label_known_by_adv = torch.randn(bz,41).requires_grad_(True)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS([im_to_recover,label_known_by_adv],lr=0.005)
    
    def compute_layer_weights(N_conv, beta, proportion_zeros):
        linear_weights = [1 + (beta - 1) * (i - 1) / (N_conv - 1) for i in range(1, N_conv + 1)]
        zero_based_weights = [1 / (1 - p) for p in proportion_zeros]
        
        convolutional_layer_weights = [l * z for l, z in zip(linear_weights, zero_based_weights)]
        fully_connected_layer_weight = np.mean(linear_weights)
        
        return convolutional_layer_weights, fully_connected_layer_weight
    
    
    N_conv = 6
    beta = 2.0 
    proportion_zeros = [0.2, 0.3, 0.1, 0.4, 0.1,0.15]  
    
    convolutional_weights, fc_weight = compute_layer_weights(N_conv, beta, proportion_zeros)

    history = []
    for i in range(0,10):
        def closure():
            optimizer.zero_grad()
            for t in range(0,10):
                # net architecture : 20*8 + 8*1 + 8 + 15
                map_vector_to_net(torch.Tensor(data_worker[t][1]),net,num_classes=41,num_dimension=70*80,model="conv")

# original gradients 
                output = net(torch.Tensor(imgs.reshape(bz,1,80,70)))
                # print(torch.argmax(output))
                y_pred = criterion(output, label_known_by_adv)
                #print(y_pred)
                original_gradient = torch.autograd.grad(y_pred, net.parameters())
                original_gradients = list((_.detach().clone() for _ in original_gradient))
                pred = net((torch.Tensor(im_to_recover.reshape(bz,1,80,70))))
                #print(pred)

                dummy_loss = criterion(pred.reshape(bz,41),label_known_by_adv)
                # print(dummy_loss)
                dummy_gradient = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                grad_diff = 0 
                for gx, gy in zip(dummy_gradient[0], original_gradients[0]): # : May need some reshaping.
                      grad_diff +=convolutional_weights*torch.sum(torch.nn.functional.cosine_similarity(original_gradients[0],dummy_gradient[0])) - 0.001*torch.abs(im_to_recover)
                      grad_diff = -grad_diff
                      print(grad_diff)

            grad_diff.backward()    
            return grad_diff
        optimizer.step(closure)
        


models = []
imgs=[]

    
    
    
plt.rcParams['figure.figsize'] = (20, 30)
f, axarr = plt.subplots(8,5)
for i in range(8):
    for j in range(4):
        axarr[i,j].imshow(im_to_recover.detach().numpy()[i*4+j], cmap='gray')
            
np.save('img_gradient_SRA.npy',im_to_recover)
 