

import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from torchvision import  transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity     
import json
from PIL import Image
class NeuralLayer(nn.Module):
    def __init__(self, input_dimension, intermediate_dimension, num_classes):
        super(NeuralLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.intermediate_dimension = intermediate_dimension
        self.fc1 = nn.Linear(input_dimension, intermediate_dimension)
        self.fc2 = nn.Linear(intermediate_dimension, num_classes)
        # self.fc3 = nn.Linear(intermediate_dimension, intermediate_dimension)

        # self.fc4 =  nn.Linear(intermediate_dimension, num_classes)
        # self.bn_256 = nn.BatchNorm1d(intermediate_dimension)

        self.relu = nn.ReLU()
    def forward(self, x):
        first_layer = self.relu(self.fc1(x))
        # second_layer = self.relu(self.fc2(first_layer))
        output = self.relu(self.fc2(first_layer))
        
        return output
    #def forward(self, x):
    #    first_layer = self.relu(self.bn_256(self.fc1(x)))
    #    second_layer = self.relu(self.bn_256(self.fc2(first_layer)))
    #    third_layer = self.relu(self.bn_256(self.fc3(second_layer)))
    #    output = self.fc4(third_layer)
    #    return output

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

net = NeuralLayer(70*80, 16, 41).to('cpu')
net_sub =NeuralLayer(80*70, 16,41).to('cpu')

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

# for worker_id in range(0,1) : 
#     with open('../LocalModelReconstructionAttacks/AT&T/logs/none/experiment_faces_bz_32_lr_0.05_lr_scheduler_constant_optimizer_sgd_fit_by_epoch_False_num_local_steps_1_precentage_attack_0.1_DP_False_epsilon_1.0/inter'+str(worker_id), 'rb') as f:
#             data_worker = json.load(f)


#     with open('/./LocalModelReconstructionAttacks/AT&T//federated_learning/data/data_faces/ORL/train/train.json', 'rb') as f:

#             reel_imgs = json.load(f)
#     begin=0        
#     bz = 32
#     label = 9
#     #label = reel_imgs["y"][begin:begin+bz]
#     imgs = torch.FloatTensor(np.array(reel_imgs["x"][begin:begin+bz]).reshape(bz,80,70))
#     reel_img = torch.Tensor(np.array(Image.open(str(label)+".jpg")))
#     im_to_recover = torch.randn(bz,80,70).requires_grad_(True)
#     label_known_by_adv = torch.randn(bz).requires_grad_(True)

#     criterion = torch.nn.CrossEntropyLoss()

#     optimizer = torch.optim.LBFGS([im_to_recover,label_known_by_adv],lr=0.0001)
    
#     history = []
#     for i in range(0,10):
#         def closure():
#             optimizer.zero_grad()
#             for t in range(180,181):
#                 # net architecture : 20*8 + 8*1 + 8 + 15
#                 map_vector_to_net(torch.Tensor(data_worker[-1][1]),net,num_classes=41,num_dimension=70*80,model="neural")

# # original gradients 
#                 output = net(torch.Tensor(imgs.reshape(bz,80*70)))
#                 # print(torch.argmax(output))
#                 y_pred = criterion(output.reshape(bz,41), label_known_by_adv)
#                 #print(y_pred)
#                 original_gradient = torch.autograd.grad(y_pred, net.parameters())
#                 original_gradients = list((_.detach().clone() for _ in original_gradient))
#                 pred = net((torch.Tensor(im_to_recover.reshape(bz,5600))))
#                 #print(pred)

#                 dummy_loss = criterion(pred.reshape(bz,41),label_known_by_adv)
#                 # print(dummy_loss)
#                 dummy_gradient = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
#                 grad_diff = 0 
#                 for gx, gy in zip(dummy_gradient[0], original_gradients[0]): # : May need some reshaping.
#                       grad_diff +=torch.sum(torch.nn.functional.cosine_similarity(original_gradients[0],dummy_gradient[0]))
#                       grad_diff = -grad_diff
#                       print(grad_diff)

#             grad_diff.backward()
#             #print(grad_diff)  
    
#             return grad_diff
#         optimizer.step(closure)


models = []
imgs=[]

    
    
for worker_id in range(0,1)    :
    # This is a toy model, for complex models run the LMRA and change the path
    results =  pd.read_json('SampleReconstructionAttackModels.json')
    models.append(results.decoded_model[0][str(worker_id)]['model'])

    map_vector_to_net(torch.Tensor(models[worker_id]),net_sub,num_classes=41,num_dimension=80*70,model="neural")
    
        
    
    imgs_recover = np.zeros((41, 80, 70))
    for label in range(41):
        img_initial = torch.zeros(80*70).requires_grad_()
        for i in range(1000):
            cost = 1 - torch.nn.functional.softmax(net_sub(img_initial), dim=0)[label]
            cost.backward()
            if i in range(0,1000): print(f"Label {label}, Iter {i}: loss {cost}")
            gradient_img = img_initial.grad.data
            img_initial.data.add_(-1, gradient_img)
            img_recovered = img_initial.reshape(80,70).detach().numpy()
            imgs_recover[label]=img_recovered
    imgs.append(imgs_recover)
    
    
    plt.rcParams['figure.figsize'] = (20, 30)
    f, axarr = plt.subplots(8,5)
    for i in range(8):
        for j in range(5):
            axarr[i,j].imshow(imgs_recover[i*5+j], cmap='gray')
            
data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        
        transforms.Normalize(mean=[0.485,],
                             std=[0.229,])
    ])       

 
orginal_images = np.load('original_images.npy')
model_based = np.load('model_based_SRA.npy')
gradient_based = np.load('gradient_based_SRA.npy')

print("Avg SSIM model-based SRA : {0:0.2f}".format((ssim(orginal_images,model_based))))
print("Avg SSIM gradient-based SRA : {0:0.2f}".format((ssim(orginal_images,gradient_based))))



 