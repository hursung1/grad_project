import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import numpy as np
import quadprog
import pyfiles.lib as lib

class FCNetwork(torch.nn.Module):
    def __init__(self, input_data_shape, hidden_layer_num, output_data_num):
        super(FCNetwork, self).__init__()
        self.num_channels, self.width, self.height = input_data_shape
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.width * self.height, hidden_layer_num//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(hidden_layer_num//2, hidden_layer_num),
            torch.nn.ReLU(hidden_layer_num),
            torch.nn.Dropout(p=0.5),
            
            torch.nn.Linear(hidden_layer_num, hidden_layer_num*2),
            torch.nn.ReLU(hidden_layer_num*2),
            torch.nn.Dropout(p=0.5),
            
            torch.nn.Linear(hidden_layer_num*2, hidden_layer_num),
            torch.nn.ReLU(hidden_layer_num),
            torch.nn.Dropout(p=0.5),
            
            torch.nn.Linear(hidden_layer_num, output_data_num)
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
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(6, 16, 7, 1), # 16 @ 56*56
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 16 @ 28*28
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(16, 16, 5, 1), # 16 @ 24*24
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 16 @ 12*12
            torch.nn.Dropout(p=0.2),
        )

        self.fc_module = torch.nn.Sequential(
            torch.nn.Linear(16*12*12, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, 7)
        )


    def forward(self, input):
        x = self.conv_module(input)
        dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, dim)
        return self.fc_module(x)
    
    
### Gradient Episodic Memory for Deep Generative Replay
class GEMLearning(torch.nn.Module):
    """
    Deep Generative Replay 
     - Solver model with Gradient Episodic Memory
    """
    def __init__(self, **kwargs):
        super(GEMLearning, self).__init__()
        self.net = kwargs['net'] # Solver Network: FCNetwork
        self.tasks = kwargs['tasks'] # Number of tasks
        self.optim = kwargs['optim'] # Optimizer
        self.criterion = kwargs['criterion'] # Criterion(CELoss, BCELoss,...)
        self.mem_size = kwargs['mem_size'] # Size of Episodic Memory
        self.num_noise = kwargs['num_noise'] # number of total train data
        self.batch_size = kwargs['batch_size'] # Batch size
        self.device = kwargs['device'] # Device (cpu or cuda)
        self.margin = kwargs['margin'] # GEM Hyperparameter(1)
        self.eps = kwargs['eps'] # GEM Hyperparameter(2)

        # Save each parameters' number of elements(numels)
        self.grad_numels = []
        for params in self.parameters():
            self.grad_numels.append(params.data.numel())

        # Make matrix for gradient w.r.t. past tasks
        self.G = torch.zeros((sum(self.grad_numels), self.tasks)).to(self.device)

        # Make matrix for accuracy w.r.t. past tasks
        self.R = torch.zeros((self.tasks, self.tasks)).to(self.device)

        print(self.optim)
        print(self.criterion)
        print("Memory size: ", self.mem_size)
        
    def train(self, data_loader, task, generator, classifier):
        self.cur_task = task
        running_loss = 0.0
        
        for i, data in enumerate(data_loader):
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)

            self.G.data.fill_(0.0)
            # Compute gradient w.r.t. past tasks with episodic memory
            ### !!!!!
            if self.cur_task > 0:
                for k in range(0, self.cur_task):
                    self.zero_grad()
                    noise = lib.sample_noise(self.mem_size, self.num_noise).to(self.device)
                    g_image = generator(noise).to(self.device)
                    g_label = classifier(g_image).max(dim=1)[1] 
                    g_pred = self.net(g_image)
                    loss = self.criterion(g_pred, g_label)
                    loss.backward()
        
                    # Copy parameters into Matrix "G"
                    j = 0
                    for params in self.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(self.grad_numels[:j])
            
                            endpt = sum(self.grad_numels[:j+1])
                            self.G[stpt:endpt, k].data.copy_(params.grad.data.view(-1))
                            j += 1
                    
            self.zero_grad()
            self.optim.zero_grad()
            
            # Compute gradient w.r.t. current continuum
            pred = self.net(x)
#             pred[:, : self.cur_task * 10].data.fill_(-10e10)
#             pred[:, (self.cur_task+1) * 10:].data.fill_(-10e10)
            
#             pred = pred[:, self.cur_task*10: (self.cur_task+1)*10]
            loss = self.criterion(pred, y)
            loss.backward()

            running_loss += loss.item()
            if i % 100 == 99:
                msg = '[%d\t%d] AVG. loss: %.3f\n'% (task+1, i+1, running_loss/100)#(i*5))
                print(msg)
                #self.log_file.write(msg)
                running_loss = 0.0
            
            if self.cur_task > 0:
                grad = []
                j = 0
                for params in self.parameters():
                    if params is not None:
                        if j == 0:
                            stpt = 0
                        else:
                            stpt = sum(self.grad_numels[:j])

                        endpt = sum(self.grad_numels[:j+1])
                        self.G[stpt:endpt, self.cur_task].data.copy_(params.grad.view(-1))
                        j += 1

                
                # Solve Quadratic Problem 
                dotprod = torch.mm(self.G[:, self.cur_task].unsqueeze(0), self.G[:, :self.cur_task+1])

                # projection
                if(dotprod < 0).sum() > 0: 
                    if i % 100 == 99:
                        print("projection")
                    mem_grad_np = self.G[:, :self.cur_task+1].cpu().t().double().numpy()
                    curtask_grad_np = self.G[:, self.cur_task].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()
                    
                    t = mem_grad_np.shape[0]
                    P = np.dot(mem_grad_np, mem_grad_np.transpose())
                    P = 0.5 * (P + P.transpose()) + np.eye(t) * self.eps
                    q = np.dot(mem_grad_np, curtask_grad_np) * (-1)
                    G = np.eye(t)
                    h = np.zeros(t) + self.margin 
                    v = quadprog.solve_qp(P, q, G, h)[0]
                    x = np.dot(v, mem_grad_np) + curtask_grad_np
                    newgrad = torch.Tensor(x).view(-1, )
    
                    # Copy gradients into params
                    j = 0
                    for params in self.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(self.grad_numels[:j])
        
                            endpt = sum(self.grad_numels[:j+1])
                            params.grad.data.copy_(newgrad[stpt:endpt].contiguous().view(params.grad.data.size()))
                            j += 1

            self.optim.step()
           
        
    def eval(self, data_loader, task):
        total = 0
        correct = 0
        self.net.eval()
        for i, data in enumerate(data_loader):
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)
                
            output = self.net(x)#[:, task * 10: (task+1) * 10]
            _, predicted = torch.max(output, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
            self.R[self.cur_task][task] = 100 * correct / total
    
    
    
### Variational AutoEncoder
class VAE(torch.nn.Module):
    def __init__(self, input_data_shape = (1, 28, 28),  hidden_layer_num = 400, latent_dim = 20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels, self.width, self.height = input_data_shape
        self.input_layer_num = self.width * self.height

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_layer_num, hidden_layer_num),
            torch.nn.ReLU()
        )

        self.z_mean_out = torch.nn.Linear(hidden_layer_num, latent_dim)
        self.log_var_out = torch.nn.Linear(hidden_layer_num, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_layer_num),
            torch.nn.ReLU(),
            
            torch.nn.Linear(hidden_layer_num, self.input_layer_num),
            torch.nn.Tanh()
        )

    def forward(self, x):
        z_mean, log_var = self.encode(x)
        latent_variable = lib.reparam_trick(z_mean, log_var)
        return self.decode(latent_variable), z_mean, log_var
    
    def encode(self, x):
        _x = x.view(-1, self.input_layer_num)
        out = self.encoder(_x)
        z_mean = self.z_mean_out(out)
        log_var = self.log_var_out(out)
        return z_mean, log_var

    def decode(self, x):
        return self.decoder(x)
    
    
### Deep Generative Replay
##### GAN
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

        if width == 128:
            conv2d_1 = torch.nn.Conv2d(in_channels=num_channels, 
                                       out_channels=width*2, 
                                       kernel_size=6, 
                                       stride=4,
                                       padding=1,
                                       bias=False)
            conv2d_2 = torch.nn.Conv2d(in_channels=width*2, 
                                       out_channels=width*4, 
                                       kernel_size=6, 
                                       stride=4,
                                       padding=1,
                                       bias=False)
            conv2d_3 = torch.nn.Conv2d(in_channels=width*4, 
                                       out_channels=output_node_size, 
                                       kernel_size=8, 
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

        elif width == 28:
            conv2d_1 = torch.nn.Conv2d(in_channels=num_channels, 
                                   out_channels=width*4, 
                                   kernel_size=4, 
                                   stride=2,
                                   padding=1,
                                   bias=False)
            conv2d_2 = torch.nn.Conv2d(in_channels=width*4, 
                                       out_channels=width*8, 
                                       kernel_size=4, 
                                       stride=2,
                                       padding=1,
                                       bias=False)
            conv2d_3 = torch.nn.Conv2d(in_channels=width*8, 
                                       out_channels=output_node_size, 
                                       kernel_size=7, 
                                       stride=1,
                                       padding=0,
                                       bias=False)

            self.network = torch.nn.Sequential(
                conv2d_1,
                torch.nn.BatchNorm2d(num_features=width*4),
                torch.nn.LeakyReLU(inplace=True),
                conv2d_2,
                torch.nn.BatchNorm2d(num_features=width*8),
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
            torch.nn.Dropout(0.5),
            fc2,
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            fc3
        )

    def forward(self, x):
        return self.network(x.view(x.shape[0], -1))
