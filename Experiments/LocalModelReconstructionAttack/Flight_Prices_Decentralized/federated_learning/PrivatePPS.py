#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:28:55 2023

@author: idriouich
"""
import numpy as np

# Define the number of clients in the network
num_clients = 10

# Define the initial values for each client
d = 1  # Dimension of theta_i
initial_theta = 10 * np.random.uniform(0, 10, (num_clients, d))
sum_x_0 = initial_theta.sum()
x = initial_theta.copy()  # Array to store updated values
initial_y = np.random.uniform(1, 1, (num_clients, d))
sum_y_0 = sum(initial_y)
avg = sum_x_0/sum_y_0
#initial_y = np.ones((num_clients, d))

weights = np.zeros((num_clients, num_clients))

# Define the number of iterations
num_iterations = 1000
# Define the directed graph adjacency matrix
adjacency_matrix = np.random.randint(0, 2, size=(num_clients, num_clients))
new_weight = np.zeros((num_clients, num_clients))
# Ensure the adjacency matrix is column stochastic
adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)

# Ensure the graph is strongly connected
for i in range(num_clients):
    if np.count_nonzero(adjacency_matrix[:, i]) == 0:
        j = np.random.randint(num_clients)
        adjacency_matrix[j, i] = 1
print(adjacency_matrix)

# Perform the algorithm
for t in range(num_iterations):
    

    for i in range(num_clients):
        p_ii = np.random.uniform(0, 1 / 2)
        for j in range(num_clients) :
            if np.count_nonzero(adjacency_matrix[:, i]) > 0:
                p_ij = (1 - p_ii) / np.count_nonzero(adjacency_matrix[:, i])
            new_weight[i,j] = p_ij
        new_weight[i,i] = p_ii

        
        # Step 2: Receive p_ij * theta_j from each agent j and sum them
        w_i = np.sum(np.matmul(new_weight,initial_theta))
        y_i =  np.sum(np.matmul(new_weight,initial_y))
        
        # Step 3: Update local parameter theta_i and execute one step of gradient descent
        epsilon = 1e-12  # Small epsilon value
        x[i, :] = w_i / (y_i + epsilon)

# Compute the average consensus value
average_consensus = np.mean(x)
print("Average Consensus:")
print(average_consensus)



