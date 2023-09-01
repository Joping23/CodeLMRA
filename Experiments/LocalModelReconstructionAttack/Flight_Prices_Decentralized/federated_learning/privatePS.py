import numpy as np

# Define the number of clients in the network
num_clients = 10

# Define the initial values for each client
d = 1  # Dimension of theta_i
initial_theta = 10 * np.random.uniform(0, 10, (num_clients, d)) 
avg = initial_theta.mean()
x = initial_theta
initial_y = np.random.uniform(0.1, 1, (num_clients, d)) 
weights = np.zeros((num_clients, d))

# Define the directed graph adjacency matrix
nodes = list(range(num_clients))
next_nodes = nodes[1:] + [nodes[0]]
adjacency_matrix = np.zeros((num_clients, num_clients), dtype=float)

# Connect each node to the next node in the circular order
for i in range(num_clients):
    adjacency_matrix[nodes[i], next_nodes[i]] = 1

# Normalize the adjacency matrix to make it column stochastic
adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)

# Define the learning rate for gradient descent
learning_rate = 0.01

# Define the number of iterations
num_iterations = 1000

# Perform the algorithm
for t in range(num_iterations):
    for i in range(num_clients):
        # Step 1: Compute p_ij * theta_i and send them to agent j
        for j in range(num_clients):
            if adjacency_matrix[j, i] != 0:
                p_ij = adjacency_matrix[j, i]
                weights[j, :] += p_ij * initial_theta[i]
        
        # Step 2: Receive p_ij * theta_j from each agent j and sum them
        w_i = np.sum(weights[:, :] * initial_theta, axis=0)
        y_i = np.sum(weights[:, :] * initial_y, axis=0)
        
        # Step 3: Update local parameter theta_i and execute one step of gradient descent
        epsilon = 1e-12  # Small epsilon value
        x[i, :] = w_i / (y_i + epsilon)
        
        # Step 4: Update private value y_i
        if len(np.nonzero(adjacency_matrix[:, i])[0]) > 0:
            p_ii = np.random.uniform(avg/2, 1/2)
            p_ji = (1 - p_ii) / len(np.nonzero(adjacency_matrix[:, i])[0])
            for j in np.nonzero(adjacency_matrix[:, i])[0]:
                weights[j, :] += p_ji * initial_theta[i] * initial_y[i]
            w_i_private = np.sum(weights[:, :] * initial_theta, axis=0)
            y_i_private = np.sum(weights[:, :] * initial_y, axis=0)
            initial_y[i] = y_i_private / (w_i_private + epsilon)
        
        # Step 5: Execute one step of gradient descent
        gradient = 2 * (x[i] - avg)
        x[i] -= learning_rate * gradient

# Compute the average consensus value
average_consensus = np.mean(x)
print("Average Consensus:")
print(average_consensus)
