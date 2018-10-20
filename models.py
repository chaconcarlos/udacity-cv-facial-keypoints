## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Dropout layer
        self.dropout1 = torch.nn.Dropout(0.4)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.dropout3 = torch.nn.Dropout(0.4)
        self.dropout4 = torch.nn.Dropout(0.4)
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale, 224x224), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (W-F)/S +1 = (224-3) / 1 + 1 = 222        
        # the output Tensor for one image, will have the dimensions: (32, 222, 222)
        # after one pool layer, this becomes (32, 111, 111)
        self.conv1 = nn.Conv2d(1, 32,  3)
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        # output size = (W-F)/S +1 = (111-3) / 1 + 1 = 109
        # the output tensor will have dimensions: (64, 109, 109)
        # after another pool layer this becomes (64, 54, 54);        
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # third conv layer: 64 inputs, 128 outputs, 2x2 conv
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # the output tensor will have dimensions: (128, 53, 53)
        # after another pool layer this becomes (128, 26, 26);  
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Fourth conv layer: 128 inputs, 256 outputs, 1x1 conv
        # output size = (W-F)/S +1 = (26-2) / 1 + 1 = 25
        # the output tensor will have dimensions: (256, 25, 25)
        # after another pool layer this becomes (256, 12, 12);  
        #self.conv4 = nn.Conv2d(128, 256, 1)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # 256 outputs * the 12 * 12 filtered/pooled map size
        # 136 output channels (2 for each of the 68 keypoint (x, y) pairs)
        
        #convolution_output_size = 256 * 12 * 12 
        convolution_output_size = 128 * 26 * 26 
        
        # First try using the information from the NaimishNet paper.
        #self.fc1 = nn.Linear(convolution_output_size, 1000)
        #self.fc2 = nn.Linear(1000, 1000)
        #self.fc3 = nn.Linear(1000, 136)
        
        # Simplifyed network, trying to reduce training time and improve accuracy.
        self.fc1 = nn.Linear(convolution_output_size, 1000)
        self.fc2 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.relu(self.conv3(x))))
        # x = self.dropout(self.pool(F.relu(self.conv4(x))))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = self.dropout4(F.relu(self.fc1(x)))
        #x = self.dropout(F.relu(self.fc2(x)))
        #x = self.fc3(x)
        
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
