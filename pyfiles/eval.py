import torch
import torchvision
import numpy as np

def eval(**kwargs):
    dataloader = kwargs['dataloader']
    num_task = kwargs['num_task']
    net = kwargs['net']
    
    each_task_acc = torch.zeros((num_task, ), dtype=int)
    if torch.cuda.is_available():
        each_task_acc = each_task_acc.cuda()

    for t in range(num_task):
        total = 0
        correct = 0
        for _, data in enumerate(dataloader[t]):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            outputs = net(x)
            _, predicted = torch.max(outputs, dim=1)
            total += x.shape[0]

            correct += (predicted == y).sum()

        each_task_acc[t] = (correct / total) * 100

    acc_mean = torch.mean(each_task_acc)
    print('[Task %d] avg accuracy: %.3f%'%(num_task+1, acc_mean))
    
    return each_task_acc, acc_mean