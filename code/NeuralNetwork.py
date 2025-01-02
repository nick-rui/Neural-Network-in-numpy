import numpy as np
from helper_functions import *

class NeuralNetwork:
    def __init__(self, input_features: int, output_features: int, num_layers=2, hidden_units=10):
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.input_features = input_features
        self.output_features = output_features
        self.weights, self.biases = initialize_parameters(num_layers, input_features, hidden_units, output_features)
    

    def predict(self, data: np.array):
        _, a_values = forward_propagation(test_data, self.weights, self.biases)
        logits = a_values[-1]
        return get_predictions(logits)    
    

    def train(self, train_data: np.array, train_labels: np.array, epochs: int, learning_rate: float):
        for epoch in range(epochs):
            z_values, a_values = forward_propagation(train_data, self.weights, self.biases)
            d_weights, d_biases = backpropagation(train_labels, z_values, a_values, self.weights)
            update_parameters(learning_rate, self.weights, self.biases, d_weights, d_biases)
    

    def get_accuracy(self, data, labels):
        _, a_values = forward_propagation(data, self.weights, self.biases)
        logits = a_values[-1]
        predictions = get_predictions(logits)
        return calculate_accuracy(predictions, labels)