

import torch
from torch import nn
from torch import optim

input_dim = 2
hidden_dim = 10
output_dim = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu") # He initialization -> He initialization is a way to initialize the weights of a neural network in a way that prevents the vanishing and exploding gradient problem-> that problem occurs when the weights of the neural network are initialized with very large or very small values
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x)) # ReLU activation which is a non-linear activation function
        x = torch.nn.functional.sigmoid(self.layer_2(x)) # Sigmoid activation function

        return x
       
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
print(model)


num_epochs = 100
loss_values = []


for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients -> zero_grad() is used to zero the gradients before the backpropagation
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)

        loss = loss_fn(pred, y.unsqueeze(-1)) # unsqueeze(-1) is used to add a dimension to the tensor. the tensor is 2D and we need to make it 3D

        loss_values.append(loss.item()) # append the loss value to the list
        loss.backward() # backpropagation
        optimizer.step() # update the weights
 
print("Training Complete")




step = range(len(loss_values))

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()