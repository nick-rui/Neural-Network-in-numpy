import pandas as pd
import numpy as np
import NeuralNetwork as nn

# Loading in Data
train_dataframe = pd.read_csv("/Users/nickrui/Documents/Neural-Network-from-scratch/Data/mnist_train.csv")
test_dataframe = pd.read_csv("/Users/nickrui/Documents/Neural-Network-from-scratch/Data/mnist_test.csv")

# Each sample is a row of the file. The first column is the label (0-9) and the next 
# 784 are greyscale pixel values (0-255)
# We orient our data into np.array with columns as each sample to be compatible with our model

train_data = train_dataframe.to_numpy()
test_data = test_dataframe.to_numpy()

train_labels = train_data[:, 0]
test_labels = test_data[:, 0]

train_data = train_data[:, 1:].T / 255.0
test_data = test_data[:, 1:].T / 255.0

input_features = len(train_data[:,0]) # 784

# Map classes to indices (This step seems trivial and unnecessary, but we continue 
# for the sake of consistency to other applications of neural networks.)

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
classes_to_idx = {0 : 0,
                  1 : 1,
                  2 : 2,
                  3 : 3,
                  4 : 4,
                  5 : 5,
                  6 : 6,
                  7 : 7,
                  8 : 8,
                  9 : 9}
num_classes = len(classes)

# Creating and Training the model:

TRAINING_CYCLES = 10
EPOCHS = 100
LEARNING_RATE = 0.1
train_accs = []
test_accs = []

model = nn.NeuralNetwork(input_features=input_features,
                        output_features=num_classes,
                        num_layers=2,
                        hidden_units=16)

for cycle in range(TRAINING_CYCLES):
    model.train(train_data=train_data, 
                train_labels=train_labels, 
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE)

    train_accuracy = model.get_accuracy(data=train_data, labels=train_labels)
    test_accuracy = model.get_accuracy(data=test_data, labels=test_labels)
    print(f"Epoch: {(cycle + 1) * EPOCHS} | Training Accuracy: {train_accuracy:.4f} | Testing Accuracy: {test_accuracy:.4f}")
    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)