import numpy as np

#define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# X is a unit vector
x = np.array([0.5, 0.1, -0.2])

# target that we want to predict
target = 0.6

#learning rate or alpha
learnrate = 0.5

#weight that we initialize
weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

#weight that produced by hidden layer as a output 
weights_hidden_output = np.array([0.1, -0.3])


# # Feed - Forward

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
print("hidden_layer_output :",hidden_layer_output)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_input)
print("Predicted output : ", output)


# # Backpropagation

## Backwards pass
## Calculating output error
error = target - output
print("error: ", error)
# Calculating error term for output layer
output_error_term = error * output * (1 - output)

# Calculating error term for hidden layer
hidden_error_term = np.dot(output_error_term, weights_hidden_output) * hidden_layer_output * (1 - hidden_layer_output)

# Calculating change in weights for hidden layer to output layer
delta_weight_hidden_output = learnrate * output_error_term * hidden_layer_output

# Calculate change in weights for input layer to hidden layer
delta_weight_input_hidden = learnrate * hidden_error_term * x[:, None]


# # Weight change between the layers

print('Change in weights for hidden layer to output layer:')
print(delta_weight_hidden_output)
print('\nChange in weights for input layer to hidden layer:')
print(delta_weight_input_hidden)

