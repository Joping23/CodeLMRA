# This document contains the steps required to reproduce the experiments in our paper.



First, we start by providing some examples on how to launch our Local Model Reconstruction Attack (LMRA). Second, we describe the steps that are required to trigger the proposed model-based Attribute Inference Attack (AIA), Source Inference Attack (SIA), and Sample Reconstruction Attack (SRA).



## Environment setup : 


The first step is to set the environment:

conda create --name my_env

conda activate my_env

conda install pip

pip install -r requirements.txt


## Datasets description : 


We consider five datasets: Flight Prices for the regression task, Adult, Leaf Synthetic, AT\&T dataset for the image classification tasks. For additional details about these datasets and how we split them in a FL scenario please refer to our paper.

For all the tabular datasets, the model is a neural network with 3 hidden layers (256 neurons per layer) followed by an ReLU activation layer. For the image datasets the model is a CNN of 6 convolutional layers followed by two fully connected layers




## Local Model Reconstruction Attacks 



Example of running the LMRA on Adult dataset, while training a neural regression model.

Go to LocalModelReconstructionAttack/Adult&Synthetic/

python3 main.py adult --model neural --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.05 --adv_lr 0.001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000

Example of running the LMRA on Flight Prices dataset, while training a neural network model.

Go to LocalModelReconstructionAttack/Flight_Prices

python3 main.py flightPrices --model neuralReg --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.00001 --adv_lr 0.0001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000

Example of running the LMRA on Leaf Synthetic dataset, while training a neural network model.

First, generate the data:

Go to LocalModelReconstructionAttack/Adult&Synthetic/ and launch:

python3 federated_learning/data/synthetic.py --num_workers 10 --dimension 100 --num_clusters 5 --num_classes 2

then launch:
python3 main.py synthetic --model neural --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.001 --adv_lr 0.001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000

Example of running the LMRA on AT&T dataset, while training a neural network model.

First split the dataset:

Go to LocalModelReconstructionAttack/AT&T and launch:

python3 federated_learning/data/data_faces/create_faces.py
Go to LocalModelReconstructionAttack/AT&T and launch:

python3 main.py faces --model neural --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.05 --adv_lr 0.001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000




Arguments of the LMRA:

•	--model: model type [ For linear regression choose ‘linearRegression’, For ‘Neural regression choose ‘neuralReg’, for logistic regression choose ‘linear’, for neural networks on classification tasks choose: ‘neural’, for convolutional neural networks on classification tasks choose: conv.
•	--bz  Batch size
•	 --num_local_steps  the selected number of local steps.
•	--device: ‘cuda’ for GPU, ‘cpu’ for CPU
•	--gnetwork_num_epochs: the gradient network Gc epochs number
•	-- adv_lr: the adversary learning rate
•	-gnetwork_features: the gradient network number of features
•	--decoded_epochs: the number of epochs to solve Equation 8.
•	--lr: the FL learning rate.
•	--num_rounds: Number of communication rounds.
•	--DP: training with differential privacy
•	--epsilon: privacy budget
•	--max_grad_norm: gradient clipping



## Attribute Inference Attacks : 


Example of launching the model-based AIA on FlightPrices:

We first need to make sure that the LMRA was previously launched, then go to the folder ExpFlightPrices and launch AIA.py:

python3 AIA.py

The procedure is similar for Adult and LeafSynthetic datasets.

Example of launching the model-based SIA on FlightPrices:

We need first to make sure that the LMRA was previously launched, then go to the folder ExpFlightPrices and launch SIA.py

## Source Inference Attacks : 


python3 SIA.py

The procedure is similar for Adult and LeafSynthetic datasets.


## Sample Reconstruction Attacks : 
Example of launching the model-based SRA on AT&T:

We first need to make sure that the LMRA was previously launched, then go to the folder ExpAT&T and launch SRA.py:

python3 SRA.py


