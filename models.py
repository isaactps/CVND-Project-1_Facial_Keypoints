## TODO: define the convolutional neural network architecture

import torch
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
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to    avoid overfitting
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 32 outputs, 3x3 conv kernel
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (32, 108, 108)
        # after another pool layer this becomes (32, 54, 54);
        self.conv2 = nn.Conv2d(32, 32, 3)
        
        # third conv layer: 32 inputs, 48 outputs, 3x3 conv kernel
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (48, 52, 52)
        # after another pool layer this becomes (48, 26, 26)
        self.conv3 = nn.Conv2d(32, 48, 3)

        # 48 outputs * the 26*26 filtered/pooled map size
        self.fc1 = nn.Linear(48*26*26, 5000)
        
        # 5000 fully connected features downsized to 1000 features
        self.fc2 = nn.Linear(5000, 1000)        
        
        # finally, create 136 output channels (for the 68 keypoints with x and y coordinates)
        self.fc3 = nn.Linear(1000, 136)

        # dropout with p=0.4
        self.fc_drop_1 = nn.Dropout(p=0.4)
        self.fc_drop_2 = nn.Dropout(p=0.6)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        # three conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc_drop_1(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop_2(x)
        x = self.fc3(x)
                
        # a modified x, having gone through all the layers of your model, should be returned
        return x
