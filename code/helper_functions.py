# Helper Functions for building our Neural Network

import numpy as np

def initialize_parameters(num_layers: int, input_features: int, hidden_units: int, output_features: int):
    '''
    Takes in integers num_layers, input_features, hidden_units, output_features to outline the structure
    of the desired neural network. Returns a tuple of two lists `weights` and `biases` where the 
    i-th element of each of them represents the weight matrix/bias vector of the (i + 1)-th row.
    '''
    weights = []
    biases = []

    for i in range(num_layers):
        rows, cols = hidden_units, hidden_units
        if i == 0:
            cols = input_features
        if i == (num_layers - 1):
            rows = output_features
        
        weights.append(np.random.rand(rows, cols) - 0.5)
        biases.append(np.random.rand(rows, 1) - 0.5)

    return weights, biases


def ReLU(X: np.array) -> np.array:
    '''
    The Rectified Linear Unit function.
    Takes in an np.array and returns an np.array with each negative element
    changed to 0 and each nonnegative element unchanged.
    '''

    return np.maximum(0, X) # operates element-wise on array x


def deriv_ReLU(X: np.array) -> np.array:
    '''
    The derivative of the Rectified Linear Unit function.
    Takes in an np.array and returns an np.array of the same size where each
    positive element is changed to a 1 and each negative element to a 0.
    '''

    return np.where(X > 0, 1, 0)


def softmax(X: np.array) -> np.array:
    '''
    Takes in an np.array and returns an np.array of the same size where
    each element is converted to a probability based on how much greater
    that element was compared to the others. The sum of elements in the 
    output array should add to 1.
    '''

    return np.exp(x) / np.sum(np.exp(x), axis=0)


def one_hot(labels: np.array) -> np.array:
    '''
    Takes in a vector np.array of labels.
    Returns a matrix np.array (2D) of size (classes, labels.size) with each
    i-th column as a "one-hot" vector where the labels[i]-th element is 1 and
    every other element 0.
    '''

    result = np.zeros(shape=[labels.max() + 1, labels.size])
    result[labels, np.arange(labels.size)] = 1
    return result


def forward_propagation(input: np.array, weights: list, biases: list):
    num_layers = len(weights)
    z_values = [None] # Pre-activation values, (input layer has no pre-activation value)
    a_values = [input] # Neuron activation values 

    for l in range(num_layers):
        W = weights[l] # Note this list is 0-indexed
        b = biases[l]  # Note this list is 0-indexed
        z = W.dot(a_values[l]) + b
        a = ReLU(z)
        z_values.append(z)
        a_values.append(a)
    
    return z_values, a_values


def backpropagation(labels: np.array, z_values: list, a_values: list, weights: list):
    n = labels.size
    y = one_hot(labels)
    num_layers = len(weights)

    # delta represents the error term in a layer
    delta_L = a_values[-1] - y # Error in the last layer
    delta_values = [delta_L] # A list of deltas to be appendeded on from the front
    for l in range(num_layers - 1, 0, -1):
        W = weights[l] # Weights of layer l + 1 (weights is 0-indexed)
        z = z_values[l] # Pre-activation values of layer l
        delta_values.insert(0, W.T.dot(delta_values[0]) * deriv_ReLU(z))
    
    # Note that delta_values[i] is the error term in the (i+1)-th layer

    d_weights = []
    d_biases = []

    for l in range(1, num_layers + 1):
        delta_l = delta_values[l - 1]
        d_weights.append((1 / n) * delta_l.dot(a_values[l - 1].T))
        d_biases.append((1 / n) * np.sum(delta_l, axis=1, keepdims=True))
    
    return d_weights, d_biases


def update_parameters(learning_rate: float, weights: list, biases: list, d_weights: list, d_biases: list):
    for i in range(len(weights)):
        weights[i] -= (learning_rate * d_weights[i])
        biases[i] -= (learning_rate * d_biases[i])

def get_predictions(logits):
  return np.argmax(logits, 0)

def calculate_accuracy(predictions, labels):
  return np.sum(predictions == labels) / labels.size

# ------------------------------------------------------------------------
# Test Cases:
# ------------------------------------------------------------------------

def test_functions():
    # Test case for MNIST dataset. Neurons per layer: 784 -> 10 -> 10
    print("Testing helper functions...")
    weights, biases = initialize_parameters(num_layers=2,
                                            input_features=784,
                                            hidden_units=10,
                                            output_features=10)
    print("Parameters successfully initialized")

    dummy_data = np.random.rand(784, 100) # 100 random inputs with values from [0,1)
    dummy_labels = np.random.randint(0, 10, size=[100, 1])

    z_values, a_values = forward_propagation(input=dummy_data,
                                            weights=weights,
                                            biases=biases)
    print("Forward propagation successful. Calculated pre-activation and activation values of all neurons.")

    d_weights, d_biases = backpropagation(labels=dummy_labels, 
                                          z_values=z_values,
                                          a_values=a_values,
                                          weights=weights)
    print("Backpropagation successful. Calculated all partial derivatives with respect to the parameters")

    learning_rate = 0.01
    update_parameters(learning_rate=learning_rate,
                      weights=weights,
                      biases=biases,
                      d_weights=d_weights,
                      d_biases=d_biases)
    print(f"Successfully updated parameters with learning rate of {learning_rate}")
    print("No errors!")

# test_functions()