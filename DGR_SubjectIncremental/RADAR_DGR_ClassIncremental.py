import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

import pyfiles.dataloader as dataloader
import pyfiles.lib as lib
import pyfiles.models as models

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='mode', type=str)
args = parser.parse_args()

data_shape = (1, 128, 128)
num_noise = batch_size = 64
epochs = 200
ld = 10
category={0: 'boxingmoving', 
          1: 'boxingstill', 
          2: 'crawling', 
          3: 'running', 
          4: 'still', 
          5: 'walking', 
          6: 'walkinglow'}

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
data_path='./data/RADAR'

### Get RADAR DATA
if args.mode == "divide_subject" or args.mode == "divide_class": # divide by subject
    print("Learning mode: %s..." % (args.mode))
    train_data, train_labels, test_data, test_labels = lib.RADARLoader(data_path, tasks, category, device, args.mode)
    tasks = train_data.shape[0]

else:
    print("Invalid input. There are only two learning modes: divide_subject, divide_class")
    sys.exit()

### Normalize data to [-1, 1]
train_data = lib.tensor_normalize(train_data)
train_labels = lib.tensor_normalize(train_labels)
test_data = lib.tensor_normalize(test_data)
test_labels = lib.tensor_normalize(test_labels)

for task in range(tasks):
    pre_gen = pre_solver = None
    ratio = 1/(task+1)
    
    ### Save previous Generator and Solver
    if task > 0:
        pre_gen = gen
        pre_solver = solver
    
    lib.model_switch_grad(pre_gen, False)
    lib.model_switch_grad(pre_solver, False)
    
    ### define new Generator, Discriminator, and Solver
    gen = models.Generator_Conv(input_node_size=num_noise, output_shape=data_shape).to(device)
    disc = models.Discriminator_Conv(input_shape=data_shape).to(device)
    
    if args.mode == "divide_class":
        solver = models.Solver(input_shape=data_shape, task+1).to(device)
    else:
        solver = models.Solver(input_shape=data_shape, 7).to(device)

    lib.init_params(gen)
    lib.init_params(disc)
    lib.init_params(solver)

    ### optimizer
    optim_g = torch.optim.Adam(gen.parameters(), lr=1e-3, betas=(0, 0.9))
    optim_d = torch.optim.Adam(disc.parameters(), lr=1e-3, betas=(0, 0.9))
    optim_s = torch.optim.Adam(solver.parameters(), lr=1e-3)

    for epoch in range(epochs):
        gen.train()
        disc.train()

        ### WGAN_GP Learning
        for i in range(10):
            for _train_data in train_data:
                x = _train_data.view(-1, 1, 128, 128).to(device)
                num_data = x.shape[0]
                noise = lib.sample_noise(num_data, num_noise).to(device)

                x_g = gen(noise)

                ### Discriminator train
                optim_d.zero_grad()

                ## Regularization Term
                eps = torch.rand(1).item()
                x_hat = (x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)).requires_grad_(True)

                loss_xhat = disc(x_hat)
                fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False).to(device)

                gradients = torch.autograd.grad(
                    outputs = loss_xhat,
                    inputs = x_hat,
                    grad_outputs=fake,
                    create_graph = True,
                    retain_graph = True,
                    only_inputs = True
                )[0]
                gradients = gradients.view(gradients.shape[0], -1)
                gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * ld

                p_real = disc(x)
                p_fake = disc(x_g.detach())

                loss_d = torch.mean(p_fake) - torch.mean(p_real) + gp
                loss_d.backward()
                optim_d.step()

                if i % 5 == 4:
                    ### Generator train
                    optim_g.zero_grad()
                    p_fake = disc(x_g)
                    loss_g = -torch.mean(p_fake)
                    loss_g.backward()
                    optim_g.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, epochs, loss_d.item(), loss_g.item()))

        if epoch % 10 == 9:
            dir_name = "imgs/"#Task_%d" % (t+1)
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            gen.eval()
            noise = lib.sample_noise(64, num_noise).to(device)

            gen_img = gen(noise)
            lib.imsave(gen_img, epoch)
            
            
        ### Solver Learning
        for i, _train_data in enumerate(train_data):
            celoss = torch.nn.CrossEntropyLoss()
            optim_s.zero_grad()
            solver.zero_grad()
            
            if pre_gen is not None:
                
            
        pass