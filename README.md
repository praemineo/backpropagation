# Backpropagation
This file contains two process :
## * backpropagation  
## * feed-forward
### For doing these process we required following :
### * Sigmoid function
### * Unit vector (X)
### * Target value that we want to predict using backpropagation
### * Learning rate (&alpha)

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
## Initialization
In initialization first we define sigmoid function
