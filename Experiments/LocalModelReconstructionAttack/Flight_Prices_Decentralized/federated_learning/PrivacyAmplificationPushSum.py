import numpy as np
import matplotlib.pyplot as plt

# Define the number of clients in the network
num_clients = 10

# Define the initial values for each client
d = 1  # Dimension of theta_i
initial_theta = 10 * np.random.uniform(0, 10, (num_clients, d))
avg = initial_theta.mean()
x = initial_theta.copy()  # Array to store updated values

# Define the number of iterations
num_iterations = 100
# Define the directed graph adjacency matrix
adjacency_matrix = np.random.randint(0, 2, size=(num_clients, num_clients))
new_weight = np.zeros((num_clients, num_clients))
# Ensure the adjacency matrix is column stochastic
adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)

# Define the privacy parameters
epsilon_values = [100, 1000,2000,3000,4000]  # Vary the epsilon values
delta = 1e-5  # Delta parameter for (epsilon, delta)-differential privacy

# Initialize arrays to store results
random_weight_cumulative_privacy_loss = []
normal_dp_cumulative_privacy_loss = []
consensus_error_values = []
consensus_error_values_random = []


# Perform the experiment
for epsilon in epsilon_values:
    random_weight_privacy_accountant = np.zeros(num_iterations)  # Privacy accountant for random weights
    normal_dp_privacy_accountant = np.zeros(num_iterations)  # Privacy accountant for normal differential privacy
    consensus_errors = np.zeros(num_clients)  # Consensus errors
    consensus_errors_random= np.zeros(num_clients)  # Consensus errors
    # Perform the algorithm with random weights
    for i in range(num_clients):
        for t in range(num_iterations):
            p_ii = np.random.uniform(0.5, 1)
            for j in range(num_clients):
                if np.count_nonzero(adjacency_matrix[:, i]) > 0:
                    p_ij = (1 - p_ii) / np.count_nonzero(adjacency_matrix[:, i])
                new_weight[i, j] = p_ij
            new_weight[i, i] = p_ii

            # Step 2: Receive p_ij * theta_j from each agent j and sum them
            w_i = np.sum(np.matmul(new_weight, initial_theta))
            y_i = np.sum(np.matmul(new_weight, np.random.uniform(1, 1, (num_clients, d))))

            # Step 3: Update local parameter theta_i and execute one step of gradient descent
            epsilon_i = epsilon / (num_clients * num_iterations)  # Privacy budget for each client at each iteration
            sigma_i = np.sqrt(2 * np.log(1.25 / delta)) / epsilon_i  # Noise standard deviation for each client at each iteration
            noise = np.random.normal(0, sigma_i, size=(d,))
            x[i, :] = (w_i / (y_i)) + noise

            # Update the privacy accountant for random weights
            random_weight_privacy_accountant[t] = np.sqrt(2 * np.log(1.25 / delta)) / epsilon_i

        # Compute the average consensus value
        average_consensus = np.mean(x)
        consensus_error = np.abs(average_consensus - avg) / avg
        consensus_errors[i] = consensus_error
    
    # Perform the algorithm with normal differential privacy
    for i in range(num_clients):
        for t in range(num_iterations):
            new_weight = adjacency_matrix
            # Step 2: Receive p_ij * theta_j from each agent j and sum them
            w_i = np.sum(np.matmul(new_weight, initial_theta))
            y_i = np.sum(np.matmul(new_weight, np.random.uniform(1, 1, (num_clients, d))))

            # Step 3: Update local parameter theta_i and execute one step of gradient descent
            epsilon_i = epsilon / (num_clients * num_iterations)  # Privacy budget for each client at each iteration
            sigma_i = np.sqrt(2 * np.log(1.25 / delta)) / epsilon_i  # Noise standard deviation for each client at each iteration
            noise = np.random.normal(0, sigma_i, size=(d,))
            x[i, :] = (w_i / (y_i)) + noise

            # Update the privacy accountant for normal differential privacy
            normal_dp_privacy_accountant[t] = np.sqrt(2 * np.log(1.25 / delta)) / epsilon_i

        # Compute the average consensus value
        average_consensus = np.mean(x)
        consensus_error = np.abs(average_consensus - avg) / avg
        consensus_errors_random[i] = consensus_error

    # Compute the cumulative privacy loss for random weights and normal differential privacy
    random_weight_cumulative_privacy_loss.append(np.sum(random_weight_privacy_accountant))
    normal_dp_cumulative_privacy_loss.append(np.sum(normal_dp_privacy_accountant))
    consensus_error_values.append(np.mean(consensus_errors))
    consensus_error_values_random.append(np.mean(consensus_errors_random))


# Plot the privacy amplification with random weights versus consensus error
plt.figure(figsize=(8, 6))
plt.scatter(random_weight_cumulative_privacy_loss, consensus_error_values_random, c=np.array(epsilon_values)/100, cmap='cool', marker='o', label='DP + Random Weights')
plt.scatter(normal_dp_cumulative_privacy_loss, consensus_error_values, c=np.array(epsilon_values)/100, cmap='cool', marker='s', label='DP')
plt.colorbar(label='Epsilon')
plt.xlabel('Cumulative Privacy Loss')
plt.ylabel('Consensus Error')
plt.title('Privacy Amplification vs. Consensus Error')
plt.legend()
plt.show()
