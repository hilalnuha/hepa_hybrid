import numpy as np

# 2. Set a global random seed for reproducibility
np.random.seed(1)

# 3. Define the structure of your MLP
input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

# Define the number of neurons in the hidden layer(s)
hidden_layer_1_size = 64

# MLP structure: input_size -> hidden_layer_1_size -> output_size
nn_structure = [input_size, hidden_layer_1_size, output_size]

# 4. Initialize the weights and biases for each layer
weights = {}
biases = {}

# Initialize weights and biases for the first layer (input to hidden)
weights['W1'] = np.random.randn(nn_structure[0], nn_structure[1]) * 0.01
biases['b1'] = np.zeros((1, nn_structure[1]))

# Initialize weights and biases for the second layer (hidden to output)
weights['W2'] = np.random.randn(nn_structure[1], nn_structure[2]) * 0.01
biases['b2'] = np.zeros((1, nn_structure[2]))

print("MLP structure defined and weights/biases initialized.")
print(f"Input size: {input_size}")
print(f"Hidden layer 1 size: {hidden_layer_1_size}")
print(f"Output size: {output_size}")
print("Weights initialized successfully.")
print("Biases initialized successfully.")
