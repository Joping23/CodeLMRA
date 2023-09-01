#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:41:59 2023

@author: idriouich
"""

import numpy as np

# Define the number of clients in the network
num_clients = 100

# Define the initial values for each client
d = 1  # Dimension of theta_i
initial_theta = 10 * np.random.uniform(0, 10, (num_clients, d))
avg = initial_theta.mean()
x = initial_theta.copy()  # Array to store updated values
initial_y = np.ones((num_clients, d))

weights = np.zeros((num_clients, num_clients))

# Define the number of iterations
num_iterations = 100
# Define the directed graph adjacency matrix
adjacency_matrix = np.random.randint(0, 2, size=(num_clients, num_clients))
# Ensure the adjacency matrix is column stochastic
adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)

# Ensure the graph is strongly connected
# for i in range(num_clients):
#     if np.count_nonzero(adjacency_matrix[:, i]) == 0:
#         j = np.random.randint(num_clients)
#         adjacency_matrix[j, i] = 1
new_weight = adjacency_matrix

# Define the privacy parameters
epsilon = 1000000.0  # Privacy budget
delta = 1e-5  # Delta parameter for (epsilon, delta)-differential privacy

# Initialize the moment accountant
gamma = np.sqrt(2 * np.log(1.25 / delta))  # Privacy parameter for Gaussian noise
alpha = 0.01  # Privacy parameter for moment computation
total_iterations = num_iterations * num_clients  # Total number of iterations
privacy_accountant = np.zeros(total_iterations)  # Privacy accountant

# Perform the algorithm
for i in range(num_clients):

    for t in range(num_iterations):

        
        # Step 2: Receive p_ij * theta_j from each agent j and sum them
        w_i = np.sum(np.matmul(new_weight, initial_theta))
        y_i = np.sum(np.matmul(new_weight, initial_y))
        
        # Step 3: Update local parameter theta_i and execute one step of gradient descent
        epsilon_i = epsilon / (num_clients * num_iterations)  # Privacy budget for each client at each iteration
        sigma_i = gamma / (epsilon_i * alpha)  # Noise standard deviation for each client at each iteration
        noise = np.random.normal(0, sigma_i, size=(d,))
        x[i, :] = (w_i / (y_i)) + noise
        
        # Update the privacy accountant
        privacy_accountant[t * num_clients + i] = privacy_accountant[max(0, t * num_clients + i - 1)] + np.sqrt(2 * np.log(1.25 / delta)) / epsilon_i

# Compute the average consensus value
average_consensus = np.mean(x)
print("Average Consensus error:")
print(abs(average_consensus - avg)/avg)

# Compute the privacy loss
privacy_loss = np.max(privacy_accountant)
print("Privacy Loss:")
print(privacy_loss)



