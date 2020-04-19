import torch
import torchvision
import numpy as np

import lib

def FineTuning(**kwargs):
    """
    Continual Learning with just Fine Tuning
    """
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    optim = kwargs['optim']
    crit = kwargs['crit']
    net = kwargs['net']

    for epoch in range(epochs):
        running_loss = 0.0
        for _, data in dataloader:
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optim.zero_grad()
            outputs = net(x)
            loss = crit(outputs, y)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        if epoch % 10 == 9:
            print("[Epoch %d] Loss: %.3f"%(epoch, running_loss))


def L2Learning(**kwargs):
    """
    Continual Learning with L2 Regularization Term
    """
    past_task_params = kwargs['past_task_params']
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    optim = kwargs['optim']
    crit = kwargs['crit']
    net = kwargs['net']
    ld = kwargs['ld']

    for epoch in range(epochs):
        running_loss = 0.0
        for _, data in dataloader:
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optim.zero_grad()
            outputs = net(x)
            loss = crit(outputs, y)

            reg = 0.0
            for past_param in past_task_params:
                for i, param in net.parameters():
                    penalty = (past_param[i] - param) ** 2
                    reg += penalty.sum()
                loss += reg * (ld / 2)

            loss.backward()
            optim.step()
            running_loss += loss.item()

        if epoch % 10 == 9:
            print("[Epoch %d] Loss: %.3f"%(epoch, running_loss))

    ### Save parameters to use next task learning
    tensor_param = []
    for params in net.parameters():
        tensor_param.append(params.detach().clone())
    tensor_param = torch.stack(tensor_param)
    past_task_params = torch.cat((past_task_params, tensor_param.unsqueeze(0)))


def EWCLearning():
    """
    Continual Learning with Fisher Regularization Term
    """
    past_task_params = kwargs['past_task_params']
    past_fisher_mat = kwargs['past_fisher_mat']
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    optim = kwargs['optim']
    crit = kwargs['crit']
    net = kwargs['net']
    ld = kwargs['ld']

    for epoch in range(epochs):
        running_loss = 0.0
        for _, data in dataloader:
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optim.zero_grad()
            outputs = net(x)
            loss = crit(outputs, y)

            reg = 0.0
            for task, past_param in enumerate(past_task_params):
                for i, param in net.parameters():
                    penalty = (past_param[i] - param) ** 2
                    penalty *= past_fisher_mat[task]
                    reg += penalty.sum()
                loss += reg * (ld / 2)

            loss.backward()
            optim.step()
            running_loss += loss.item()

        if epoch % 10 == 9:
            print("[Epoch %d] Loss: %.3f"%(epoch, running_loss))

    ### Save parameters to use at next task learning
    tensor_param = []
    for params in net.parameters():
        tensor_param.append(params.detach().clone())
    tensor_param = torch.stack(tensor_param)
    past_task_params = torch.cat((past_task_params, tensor_param.unsqueeze(0)))

    ### Save Fisher matrix
    FisherMatrix = lib.get_fisher(net, crit, dataloader)
    past_fisher_mat = torch.cat((past_fisher_mat, FisherMatrix.unsqueeze(0)))
    
    
def DGRLearning(**kwargs):
    TrainDataLoaders = kwargs['TrainDataLoaders']
    TestDataLoaders = kwargs['TestDataLoaders']
    batch_size = kwargs['batch_size']
    num_noise = kwargs['num_noise']
    cur_task = kwargs['cur_task']
    gen = kwargs['gen']
    disc = kwargs['disc']
    solver = kwargs['solver']
    pre_gen = kwargs['pre_gen']
    pre_solver = kwargs['pre_solver']
    ratio = kwargs['ratio']
    epochs = kwargs['epochs']
    
    
    assert (ratio >=0 or ratio <= 1)

    ld = 10
    optim_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0, 0.9))
    optim_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0, 0.9))
    optim_s = torch.optim.Adam(solver.parameters(), lr=0.001)
    TrainDataLoader = TrainDataLoaders[cur_task]
    
    # Generator Training
    for epoch in range(epochs):
        gen.train()
        disc.train()
        for i, (x, y) in enumerate(TrainDataLoader):
            x = x.view(-1, 28*28)
            num_data = x.shape[0]
            noise = lib.sample_noise(num_data, num_noise)
            
            if torch.cuda.is_available():
                x = x.cuda()
                noise = noise.cuda()
                 
            if pre_gen is not None:
                with torch.no_grad():
                    # append generated image & label from previous scholar
                    x_g = pre_gen(lib.sample_noise(batch_size, num_noise))
                    x = torch.cat((x, x_g))                    
                    perm = torch.randperm(x.shape[0])[:num_data]
                    x = x[perm]
                
            #x = x.unsqueeze(1)
            
            ### Discriminator train
            optim_d.zero_grad()
            x_g = gen(noise)

            ## Regularization term
            eps = torch.rand(1).item()
            x_hat = (x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)).requires_grad_(True)

            loss_xhat = disc(x_hat)
            fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False)
            if torch.cuda.is_available():
                fake = fake.cuda()
                
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
            
            #if i % 5 == 4:
            ### Generator Training
            optim_g.zero_grad()
            p_fake = disc(x_g)

            loss_g = -torch.mean(p_fake)
            loss_g.backward()
            optim_g.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, epochs, loss_d.item(), loss_g.item()))
        if epoch % 10 == 9:
            gen_image = gen(lib.sample_noise(24, num_noise)).view(24, 1, 28, 28)
            lib.imshow_grid(gen_image)
    
    # train solver
    for image, label in TrainDataLoader:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        output = solver(image)
        loss = celoss(output, label) * ratio

        if pre_solver is not None:
            noise = lib.sample_noise(batch_size, num_noise)
            g_image = pre_gen(noise)
            g_label = pre_solver(g_image).max(dim=1)[1]
            g_output = solver(g_image)
            loss += celoss(g_output, g_label) * (1 - ratio)

        loss.backward()
        optim_s.step()