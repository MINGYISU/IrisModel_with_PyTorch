import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a Model Class that ingerits nn.Module
class IrisModel(nn.Module):
    # Input Layer (4 features of the flower) -> 
    # Hidden Layer1 (number of neurons) -> 
    # H2 (n) -> 
    # Output Layer (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=4, h2=8, h3=16, out_features=3):
        super().__init__() # initialize the parent class nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # fc1(x) means it is implementing fc1's __call__ function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x
    

# Pick a manual seed for randomization
torch.manual_seed(41)
# create a model
model = IrisModel()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# Replace string values (the 3 output classes) with numerical values
# Setosa -> 0.0
# Versicolor -> 1.0
# Virginica -> 2.0
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)

# train Test Split set X, y
X = my_df.drop('species', axis=1)
y = my_df['species']

# Convert these to numpy arrays
X = X.to_numpy(dtype=np.float32) # this is ndarray
y = y.to_numpy(dtype=np.float16)

from sklearn.model_selection import train_test_split

# Train Test Split: 0.2 means 20% of the data will be used for testing
# random_state is a seed value to ensure reproducibility
# 41 is the seed value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert the X features to PyTorch FloatTensors, floattensor since we have float values
X_test = torch.from_numpy(X_test)
X_train = torch.from_numpy(X_train)

# Convert the y labels to PyTorch LongTensors
y_test = torch.from_numpy(y_test).long()
y_train = torch.from_numpy(y_train).long()

# Set the criterion of model to measure the error, how far off the prediction are from
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
# learning rate is how much the model adjusts its weights
# for each iteration (epoch), the data goes through the model from the input layer all the way to the output layer, to adjust the weights, and by repeating this process, ie, letting the data go through the model again and again to find the best weights
# paramerters return the model's para, ie, fc1, fc2, and out
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train our model
# Epochs: how many times we want to data to run through the network
epochs = 100
losses = []
for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train) # get predicted results

    # measure the loss/error, gonna be high at first since all the weights are random assigned at first
    loss = criterion(y_pred, y_train) # y_pred vs y_train, predicted values and trained values

    # keep track of our losses
    losses.append(loss.detach().numpy())

    # print every 10 epochs
    if i % 10 == 0 or i == epochs - 1:
        print(f'Epoch: {i} & Loss: {loss}')

    # Do some back propagation: take the error rate of forward propagation and 
    #   feed it back through the network to fine tune the weights
    optimizer.zero_grad() # set the gradients equal to zero
    loss.backward() # backpropagate the loss
    optimizer.step() # update the weights

# graph it
plt.plot(range(epochs), losses)
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.show()
