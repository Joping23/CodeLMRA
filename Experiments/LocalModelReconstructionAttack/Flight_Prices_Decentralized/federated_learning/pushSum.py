import numpy as np

# Define the number of clients in the network
num_clients = 10

# Define the initial values for each client
d = 1  # Dimension of theta_i
initial_theta = 100 * np.random.randn(num_clients, d)
avg = initial_theta.mean()
initial_y = np.ones((num_clients, d))
weights = np.zeros((num_clients, d))

# Define the directed graph adjacency matrix
adjacency_matrix = np.array([
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

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
        epsilon = 1e-4  # Small epsilon value
        initial_theta[i, :] = w_i / (y_i + epsilon)

# Compute the average consensus value
average_consensus = np.mean(initial_theta)
print("Average Consensus:")
print(average_consensus)
