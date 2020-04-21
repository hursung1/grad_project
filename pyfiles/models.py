import torch
import numpy as np

class FCNetwork(torch.nn.Module):
    def __init__(self, hidden_layer_num):
        super(FCNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28*28, hidden_layer_num),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_layer_num, hidden_layer_num),
            torch.nn.ReLU(hidden_layer_num),

            torch.nn.Linear(hidden_layer_num, 10)
        )

    def forward(self, x):
        _x = x.view(x.shape[0], -1)
        return self.net(_x)

    
class ConvolutionNetwork(torch.nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        self.conv_module = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5, 1), # 6 @ 124*124
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 6 @ 62*62
            torch.nn.Conv2d(6, 16, 7, 1), # 16 @ 56*56
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 16 @ 28*28
            # Conv 추가
            torch.nn.Conv2d(16, 16, 5, 1), # 16 @ 24*24
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2) # 16 @ 12*12
        )

        self.fc_module = torch.nn.Sequential(
            torch.nn.Linear(16*12*12, 128),
            torch.nn.ReLU(),
            #Linear 빼고
            #dropout 추가
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, 7)
        )


    def forward(self, input):
        x = self.conv_module(input)
        dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, dim)
        return self.fc_module(x)
    

### GAN
class Generator_Conv(torch.nn.Module):
    """
    Generator Class for GAN
    """
    def __init__(self, input_node_size, output_shape, hidden_node_size=256,
                    hidden_node_num=3):
        """
        input_node_size: dimension of latent vector
        output_shape: dimension of output image
        """
        super(Generator_Conv, self).__init__()

        self.input_node_size = input_node_size
        self.output_shape = output_shape
        num_channels, width, _ = output_shape

        layer_channels = []
#         if width <= 32:
        layer_channels.append(width//2)
        layer_channels.append(width//4)

        conv2d_1 = torch.nn.ConvTranspose2d(in_channels=input_node_size,
                                   out_channels=width*4, 
                                   kernel_size=layer_channels[1], 
                                   stride=1,
                                   padding=0,
                                   bias=False)
        conv2d_2 = torch.nn.ConvTranspose2d(in_channels=width*4, 
                                   out_channels=width*2, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.ConvTranspose2d(in_channels=width*2, 
                                   out_channels=num_channels, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features = width*4),
            torch.nn.ReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features = width*2),
            torch.nn.ReLU(inplace=True),
            conv2d_3,
            torch.nn.Tanh()
        )

    def forward(self, x):
        _x = x.view(-1, self.input_node_size, 1, 1)
        return self.network(_x)
    
    
class Discriminator_Conv(torch.nn.Module):
    """
    Discriminator Class for GAN
    """
    def __init__(self, input_shape, hidden_node_size=256, output_node_size=1):
        """
        Parameters
        ----------
        input_shape: (C, W, H)

        """
        super(Discriminator_Conv, self).__init__()
        num_channels, width, _ = input_shape

        conv2d_1 = torch.nn.Conv2d(in_channels=num_channels, 
                                   out_channels=width*2, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_2 = torch.nn.Conv2d(in_channels=width*2, 
                                   out_channels=width*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
        conv2d_3 = torch.nn.Conv2d(in_channels=width*4, 
                                   out_channels=output_node_size, 
                                   kernel_size=7, 
                                   stride=1,
                                   padding=0,
                                   bias=False)

        self.network = torch.nn.Sequential(
            conv2d_1,
            torch.nn.BatchNorm2d(num_features=width*2),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_2,
            torch.nn.BatchNorm2d(num_features=width*4),
            torch.nn.LeakyReLU(inplace=True),
            conv2d_3,
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)

    
class Solver(torch.nn.Module):
    """
    Solver Class for Deep Generative Replay
    """
    def __init__(self, input_data_shape, T_n):
        assert len(input_data_shape) == 3

        super(Solver, self).__init__()
        num_channels, width, height = input_data_shape
        fc1 = torch.nn.Linear(num_channels*width*height, 128)
        fc2 = torch.nn.Linear(128, 256)
        fc3 = torch.nn.Linear(256, T_n)
        self.network = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.ReLU(),
            fc3
        )

    def forward(self, x):
        return self.network(x.view(x.shape[0], -1))
