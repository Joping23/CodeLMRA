This document contains the steps required to reproduce the experiments in our paper.
First, we start by providing some examples on how to launch our Local Model Reconstruction Attack (LMRA). Second, we describe the steps that are required to trigger the proposed model-based Attribute Inference Attack (AIA), Source Inference Attack (SIA), and Sample Reconstruction Attack (SRA).
The first step is to set the environment:
conda create --name my_env
conda activate my_env
conda install pip
pip install -r requirements.txt
Example of running the LMRA on Adult dataset, while training a neural regression model.
Go to LocalModelReconstructionAttack/Adult&Synthetic/
python main.py adult --model neural --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.01 --adv_lr 0.001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000
Example of running the LMRA on Flight Prices dataset, while training a neural network model.
Go to LocalModelReconstructionAttack/Flight_Prices
python main.py flightPrices --model neuralReg --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.0001 --adv_lr 0.001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000
Example of running the LMRA on Leaf Synthetic dataset, while training a neural network model.
First, generate the data:
Go to LocalModelReconstructionAttack/Adult&Synthetic/ and launch:
python federated_learning/data/synthetic.py --num_workers 10 --dimension 100 --num_clusters 5 --num_classes 2
then launch:
python main.py synthetic --model neural --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.001 --adv_lr 0.001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000
Example of running the LMRA on AT&T dataset, while training a neural network model.
First split the dataset:
Go to LocalModelReconstructionAttack/AT&T and launch:
Python federated_learning/data/data_faces/create_faces.py
Go to LocalModelReconstructionAttack/AT&T and launch:
python main.py faces --model neural --num_workers 10 --num_rounds 1000 --bz 256 --num_local_steps 1 --device "cuda" --gnetwork_num_epochs 10000 --num_trials_to_decode 1 --lr 0.05 --adv_lr 0.001 --gnetwork_features 1000 --start_point global_model --decoded_epochs 20000
Example of launching the model-based AIA on FlightPrices:
We first need to make sure that the LMRA was previously launched, then go to the folder ExpFlightPrices and launch AIA.py:
python AIA.py
The procedure is similar for Adult and LeafSynthetic datasets.
Example of launching the model-based SIA on FlightPrices:
We need first to make sure that the LMRA was previously launched, then go to the folder ExpFlightPrices and launch SIA.py
python SIA.py
The procedure is similar for Adult and LeafSynthetic datasets.
Example of launching the model-based SRA on AT&T:
We first need to make sure that the LMRA was previously launched, then go to the folder ExpAT&T and launch SRA.py:
python SRA.py
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
•	--max_grad_norm: gradient clipping# CodeLMRA
