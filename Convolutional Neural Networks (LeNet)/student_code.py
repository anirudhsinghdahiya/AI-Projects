# python imports
import os
from tqdm import tqdm
import numpy

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()

        # Initialize the first convolutional layer
        # in_channels=3: Input image has 3 color channels (RGB)
        # out_channels=6: This layer will create 6 feature maps
        # kernel_size=5: Use a 5x5 filter to convolve around the input image
        # stride=1: The filter moves 1 pixel at a time
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)

        # Initialize the first pooling layer
        # kernel_size=2: Use a 2x2 window to perform max pooling
        # stride=2: The window moves 2 pixels at a time (non-overlapping window)
        self.pool_layer_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Initialize the second convolutional layer
        # in_channels=6: This layer takes the 6 feature maps from the previous layer as input
        # out_channels=16: This layer will create 16 feature maps
        self.conv_layer_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Initialize the second pooling layer, similar to the first pooling layer
        self.pool_layer_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Initialize the first fully connected (dense) layer
        # in_features=16*5*5: Flattened feature maps from the previous layer become the input
        # out_features=256: This layer has 256 neurons
        self.dense_layer_1 = nn.Linear(in_features=16*5*5, out_features=256)

        # Initialize the second fully connected layer
        # in_features=256: Takes the output of the previous dense layer
        # out_features=128: This layer has 128 neurons
        self.dense_layer_2 = nn.Linear(in_features=256, out_features=128)

        # Initialize the output layer
        # in_features=128: Takes the output of the previous dense layer
        # out_features=num_classes: The final output size corresponds to the number of classes
        self.output_layer = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # A dictionary to hold the shape of the tensor at different layers
        layer_shapes = {}

        # Pass the input through the first convolutional layer, then apply ReLU activation function,
        # and finally through the first max pooling layer
        x = self.pool_layer_1(F.relu(self.conv_layer_1(x)))
        layer_shapes[1] = list(x.size())  # Record the tensor shape after the first pooling layer

        # Pass the result through the second convolutional layer, apply ReLU,
        # and then through the second max pooling layer
        x = self.pool_layer_2(F.relu(self.conv_layer_2(x)))
        layer_shapes[2] = list(x.size())  # Record the tensor shape after the second pooling layer

        # Flatten the tensor to prepare it for the dense layers
        x = x.view(-1, 16*5*5)  # The '-1' infers the batch size
        layer_shapes[3] = list(x.size())  # Record the tensor shape after flattening

        # Pass the flattened tensor through the first dense layer, then apply ReLU
        x = F.relu(self.dense_layer_1(x))
        layer_shapes[4] = list(x.size())  # Record the tensor shape after the first dense layer

        # Repeat for the second dense layer
        x = F.relu(self.dense_layer_2(x))
        layer_shapes[5] = list(x.size())  # Record the tensor shape after the second dense layer

        # Pass the result through the output layer
        x = self.output_layer(x)
        layer_shapes[6] = list(x.size())  # Record the tensor shape after the output layer

        # Return the final output and the dictionary containing tensor shapes at various layers
        return x, layer_shapes



def count_model_params():
    # Initialize the LeNet model to access its parameters
    model = LeNet()
    
    # Variable to hold the total count of parameters
    model_params = 0

    # Loop through each parameter in the model
    for _, param in model.named_parameters():
        # Each 'param' is a tensor containing the parameters of one layer
        # 'numel()' returns the total number of elements in the tensor, 
        # which corresponds to the number of parameters in the layer
        model_params += param.numel()

    # The total number of parameters is typically very large, so for readability,
    # it's divided by 1 million ('1e6') to represent the value in millions
    return model_params / 1e6




def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc



