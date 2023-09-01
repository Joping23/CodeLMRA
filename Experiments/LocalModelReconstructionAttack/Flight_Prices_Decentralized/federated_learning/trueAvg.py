#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:17:07 2023

@author: idriouich
"""

import numpy as np

# Define the number of clients in the network
num_clients = 100

# Define the initial values for each client
d = 1  # Dimension of theta_i
initial_theta = 10 * np.random.uniform(0,10,(num_clients, d))
avg = initial_theta.mean()
x = initial_theta
initial_y = np.ones((num_clients, d))
weights = np.zeros((num_clients, d))

# Define the directed graph adjacency matrix
adjacency_matrix = np.zeros((num_clients, num_clients))
for i in range(num_clients - 1):
    adjacency_matrix[i, i+1] = 1
adjacency_matrix[num_clients-1, 0] = 1

# Normalize the adjacency matrix to make it column stochastic
adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)

# Define the learning rate for gradient descent
learning_rate = 0.01

# Define the number of iterations
num_iterations = 10000

# Perform the algorithm
for t in range(num_iterations):

        
    # Step 2: Receive p_ij * theta_j from each agent j and sum them
    w_i = np.sum(np.matmul(adjacency_matrix,initial_theta))
    y_i = np.sum(np.matmul(adjacency_matrix,initial_y))
    
    # Step 3: Update local parameter theta_i and execute one step of gradient descent
    epsilon = 0  # Small epsilon value
    x[i, :] = w_i / (y_i + epsilon)

# Compute the average consensus value
average_consensus = np.mean(x)
print("Average Consensus:")
print(average_consensus)
