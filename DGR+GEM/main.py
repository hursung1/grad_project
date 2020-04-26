import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse

import pyfiles.dataloader as datasets
import pyfiles.models as models
import pyfiles.lib as lib
# import pyfiles.train as train

parser = argparse.ArgumentParser()
parser.add_argument("epmemsize", type=int, help="Episodic memory size")
args = parser.parse_args()

batch_size = num_noise = 64

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

TrainDataLoaders = []
TestDataLoaders  = []

for i in range(5):
    # MNIST dataset
    if i < 4:
        shuffle_seed = np.arange(28*28)
        np.random.shuffle(shuffle_seed)
        
        TrainDataSet = datasets.MNIST_PermutedDataset(source='../data/', 
                                                train=True, 
                                                transform=transform, 
                                                shuffle_seed = shuffle_seed,
                                                download=True, )

        TestDataSet = datasets.MNIST_PermutedDataset(source='../data/', 
                                               train=False, 
                                               transform=transform, 
                                               shuffle_seed = shuffle_seed,
                                               download=True, )

    else:
        TrainDataSet = torchvision.datasets.MNIST(
                                root='../data',
                                train=True,
                                transform=transform,
                                download=True)
        TestDataSet = torchvision.datasets.MNIST(
                                root='../data',
                                train=False,
                                transform=transform,
                                download=True)
        
    TrainDataLoaders.append(torch.utils.data.DataLoader(TrainDataSet, 
                                                        batch_size=batch_size, 
                                                        shuffle=True,
                                                        num_workers=2))
    TestDataLoaders.append(torch.utils.data.DataLoader(TestDataSet, 
                                                       batch_size=batch_size, 
                                                       shuffle=False,
                                                       num_workers=2))

#====== Hyperparameters =======
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
ld = 10
num_tasks = 5
gen_epochs = 200
# solver_epochs = 40

#====== Accuracy list & dictionary ======
gan_p_real_list = []
gan_p_fake_list = []
solver_acc_dict = {'total':{}}


#====== Set solver =======
solver_network = models.Solver((1, 28, 28), 10).to(device)
celoss = torch.nn.CrossEntropyLoss().to(device)
optim_s = torch.optim.Adam(solver_network.parameters(), lr=0.001)

solver = models.GEMLearning(
    net = solver_network,
    tasks = num_tasks,
    optim = optim_s,
    criterion = celoss,
    mem_size = args.epmemsize,
    num_noise = num_noise,
    batch_size = batch_size,
    device = device,
    margin = 0.5,
    eps = 0.001,
)

#======= Training ========
for t in range(num_tasks):
    ratio = 1 / (t+1) # current task's ratio 
    solver_acc_dict[t] = {}
    pre_gen = None
    pre_solver = None
    if t > 0:
        pre_gen = gen
        pre_solver = deepcopy(solver.net)

        lib.model_grad_switch(pre_gen, False)
        lib.model_grad_switch(pre_solver, False)

    gen = models.Generator_Conv(input_node_size=num_noise, output_shape=(1, 28, 28), hidden_node_size=256,hidden_node_num=2).to(device)
    disc = models.Discriminator_Conv(input_shape=(1, 28, 28)).to(device)
    
    lib.init_params(gen)
    lib.init_params(disc)
    
    TrainDataLoader = TrainDataLoaders[t]

    optim_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0, 0.9))
    optim_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0, 0.9))

    # Generator Training
    for epoch in range(gen_epochs):
        gen.train()
        disc.train()

        for i, (x, _) in enumerate(TrainDataLoader):
            x = x.to(device)
            num_data = x.shape[0]
            noise = lib.sample_noise(num_data, num_noise, device).to(device)

            if pre_gen is not None:
                with torch.no_grad():
                    # append generated image & label from previous scholar
                    datapart = int(num_data*ratio)
                    perm = torch.randperm(num_data)[:datapart]
                    x = x[perm]

                    x_g = pre_gen(lib.sample_noise(num_data, num_noise, device))
                    perm = torch.randperm(num_data)[:num_data - datapart]
                    x_g = x_g[perm]

                    x = torch.cat((x, x_g))
                
            ### Discriminator train
            optim_d.zero_grad()
            disc.zero_grad()
            x_g = gen(noise)

            ## Regularization term
            eps = torch.rand(1).item()
            x_hat = x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)
            x_hat.requires_grad = True

            loss_xhat = disc(x_hat)
            fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False)
            if torch.cuda.is_available():
                fake = fake.to(device)
                
            gradients = torch.autograd.grad(outputs = loss_xhat,
                                            inputs = x_hat,
                                            grad_outputs=fake,
                                            create_graph = True,
                                            retain_graph = True,
                                            only_inputs = True)[0]
            gradients = gradients.view(gradients.shape[0], -1)
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * ld
            
            p_real = disc(x)
            p_fake = disc(x_g.detach())

            loss_d = torch.mean(p_fake) - torch.mean(p_real) + gp
            loss_d.backward()
            optim_d.step()

            ### Generator Training
            if i % 5 == 4:
                gen.zero_grad()
                optim_g.zero_grad()
                p_fake = disc(x_g)

                loss_g = -torch.mean(p_fake)
                loss_g.backward()
                optim_g.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, gen_epochs, loss_d.item(), loss_g.item()))
        if epoch % 100 == 99:
            dir_name = "imgs/Task_%d" % (t+1)
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            noise = lib.sample_noise(64, num_noise, device)
            gen_image = gen(noise)
            torchvision.utils.save_image(gen_image, 'imgs/Task_%d/%03d.png'%(t+1, epoch+1))
            
    #======= Train Solver ======
    solver.train(TrainDataLoader, t, gen, pre_solver)

    #====== Evaluation ====== 
    accuracy_list = []
    for i, testdataloader in enumerate(TestDataLoaders[:t+1]):
        solver.eval(testdataloader, i)

    print(solver.R)
    
    
acc_dict = {}
for i, acc_per_task in enumerate(solver.R):
    solver_acc_dict[i] = acc_per_task.tolist()
    acc_sum = torch.sum(acc_per_task).item()
    avg_acc = acc_sum/(i+1)
    acc_dict[i+1] = avg_acc
    
solver_acc_dict['total'] = acc_dict
print(solver_acc_dict)

plt.xlabel('task #')
plt.ylabel('accuracy')
for i in range(num_tasks):
    plt.axvline(x=i, color='darkgrey', linestyle=':')
    
x, y = list(solver_acc_dict['total'].keys()), list(solver_acc_dict['total'].values())
plt.plot(x, y)
plt.savefig('result.png', dpi=300)