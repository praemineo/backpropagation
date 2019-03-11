{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Initialization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "#define sigmoid function\n",
    "def sigmoid(x):\n",
    "    return 1 / (1 + np.exp(-x))\n",
    "\n",
    "# X is a unit vector\n",
    "x = np.array([0.5, 0.1, -0.2])\n",
    "\n",
    "# target that we want to predict\n",
    "target = 0.6\n",
    "\n",
    "#learning rate or alpha\n",
    "learnrate = 0.5\n",
    "\n",
    "#weight that we initialize\n",
    "weights_input_hidden = np.array([[0.5, -0.6],\n",
    "                                 [0.1, -0.2],\n",
    "                                 [0.1, 0.7]])\n",
    "\n",
    "#weight that produced by hidden layer as a output \n",
    "weights_hidden_output = np.array([0.1, -0.3])"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Feed - Forward"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hidden_layer_output : [0.55971365 0.38698582]\n",
      "Predicted output :  0.48497343084992534\n"
     ]
    }
   ],
   "source": [
    "## Forward pass\n",
    "hidden_layer_input = np.dot(x, weights_input_hidden)\n",
    "hidden_layer_output = sigmoid(hidden_layer_input)\n",
    "print(\"hidden_layer_output :\",hidden_layer_output)\n",
    "\n",
    "output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)\n",
    "output = sigmoid(output_layer_input)\n",
    "print(\"Predicted output : \", output)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Backpropagation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "error:  0.11502656915007464\n"
     ]
    }
   ],
   "source": [
    "## Backwards pass\n",
    "## Calculating output error\n",
    "error = target - output\n",
    "print(\"error: \", error)\n",
    "# Calculating error term for output layer\n",
    "output_error_term = error * output * (1 - output)\n",
    "\n",
    "# Calculating error term for hidden layer\n",
    "hidden_error_term = np.dot(output_error_term, weights_hidden_output) * hidden_layer_output * (1 - hidden_layer_output)\n",
    "\n",
    "# Calculating change in weights for hidden layer to output layer\n",
    "delta_weight_hidden_output = learnrate * output_error_term * hidden_layer_output\n",
    "\n",
    "# Calculate change in weights for input layer to hidden layer\n",
    "delta_weight_input_hidden = learnrate * hidden_error_term * x[:, None]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Weight change between the layers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Change in weights for hidden layer to output layer:\n",
      "[0.00804047 0.00555918]\n",
      "\n",
      "Change in weights for input layer to hidden layer:\n",
      "[[ 1.77005547e-04 -5.11178506e-04]\n",
      " [ 3.54011093e-05 -1.02235701e-04]\n",
      " [-7.08022187e-05  2.04471402e-04]]\n"
     ]
    }
   ],
   "source": [
    "print('Change in weights for hidden layer to output layer:')\n",
    "print(delta_weight_hidden_output)\n",
    "print('\\nChange in weights for input layer to hidden layer:')\n",
    "print(delta_weight_input_hidden)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
