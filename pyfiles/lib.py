import os
import torch
import numpy as np
import scipy.misc
import torchvision
from PIL import Image

def setPMNISTDataLoader(num_task, batch_size):
    """

    Returns
    -------------
    List of Permuted MNIST train/test Dataloaders
    """
    train_loader = []
    test_loader = []

    for i in range(num_task):
        shuffle_seed = np.arange(28*28)
        np.random.shuffle(shuffle_seed)
        train_loader[i] = torch.utils.data.DataLoader(
            PermutedMNISTDataLoader(
                train=True, 
                shuffle_seed=shuffle_seed),
            batch_size=batch_size)
        
        test_loader[i] = torch.utils.data.DataLoader(
            PermutedMNISTDataLoader(
                train=False, 
                shuffle_seed=shuffle_seed),
            batch_size=batch_size)
    
    return train_loader, test_loader


def get_fisher(net, crit, dataloader):
    FisherMatrix = []
    for params in net.parameters():
        if params.requires_grad:
            ZeroMat = torch.zeros_like(params)
            FisherMatrix.append(ZeroMat)

    FisherMatrix = torch.stack(FisherMatrix)

    for data in enumerate(dataloader):
        x, y = data
        num_data = x.shape[0]
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        net.zero_grad()
        outputs = net(x)
        loss = crit(outputs, y)
        loss.backward()

        for i, params in enumerate(net.parameters()):
            if params.requires_grad:
                FisherMatrix[i] += params.grad.data ** 2 / num_data

    return FisherMatrix


def RADARLoader(root, category, device, learn_mode="divide_subject"):
    """
    returns RADAR Dataset
    default: divide by subject (Data: (# of subject, # of classes, # of data, width, height))
    """
    train_data = []
    test_data = []
    subjects = 12

    for task in range(1, subjects+1):
        data_path = root + '/Subject%d'%(task)

        train_data_per_task = []
        test_data_per_task = []
        for _, cat in category.items():
            # train data
            train_data_per_cat = []
            test_data_per_cat = []

            for i in range(1, 10):
                filename = 'Human_Spect_test%d_%s_0%d.png'%(task, cat, i)
                file_path = os.path.join(data_path, 'train', cat, filename)
                img = Image.open(file_path)
                img.load()
                img = np.array(img)
                train_data_per_cat.append(img)

            # test data
            for i in range(10, 13):
                filename = 'Human_Spect_test%d_%s_%d.png'%(task, cat, i)
                file_path = os.path.join(data_path, 'test', cat, filename)
                img = Image.open(file_path)
                img.load()
                img = np.array(img)
                test_data_per_cat.append(img)

            train_data_per_task.append(train_data_per_cat)
            test_data_per_task.append(test_data_per_cat)

        train_data.append(train_data_per_task)
        test_data.append(test_data_per_task)

    train_data = torch.Tensor(train_data).to(device)
    test_data = torch.Tensor(test_data).to(device)

    
    if learn_mode == "divide_class":
        train_data = train_data.permute(1,0,2,3,4) 
        test_data = test_data.permute(1,0,2,3,4)
        
    print("Train Data Shape: ", train_data.shape)
    print("Test Data Shape: ", test_data.shape)

    train_labels = np.zeros((9, ))
    test_labels = np.zeros((3, ))

    for i in range(1, 7):
        train_labels = np.vstack((train_labels, np.zeros((9, )) + i))
        test_labels = np.vstack((test_labels, np.zeros((3, )) + i))

    train_labels = torch.Tensor(train_labels).type(torch.LongTensor).to(device)
    test_labels = torch.Tensor(test_labels).type(torch.LongTensor).to(device)

    print("Train Label Shape: ", train_labels.shape)
    print("Test Label Shape: ", test_labels.shape)
    
    return train_data, train_labels, test_data, test_labels


def imsave(img, epoch, path='imgs'):
    assert type(img) is torch.Tensor
    if not os.path.isdir(path):
        os.mkdir(path)
        
    fig = torchvision.utils.make_grid(img.cpu().detach()).numpy()[0]
    scipy.misc.imsave(path+'/%03d.png'%(epoch+1), fig)
    

def sample_noise(batch_size, N_noise, device='cpu'):
    """
    Returns 
    """
    return torch.randn(batch_size, N_noise).to(device)


def init_params(model):
    """
    initiallize network's parameter
    """
    for p in model.parameters():
        if(p.dim() > 1):
            torch.nn.init.xavier_normal_(p)
        else:
            torch.nn.init.uniform_(p, 0.1, 0.2)
            
            
def tensor_normalize(tensor):
    """
    Normalize tensor to [-1, 1]
    """
    _tensor = tensor.detach().clone()
    _tensor_each_sum = _tensor.sum(dim=1)
    _tensor /= _tensor_each_sum.unsqueeze(1)

    _tensor[torch.isnan(_tensor)] = 0.0
    _tensor = 2*_tensor - 1
    return _tensor


def model_grad_switch(net, requires_grad):
    """
    switch network's requires_grad
    """
    for params in net.parameters():
        params.requires_grad_(requires_grad)
        
        
def solver_evaluate(cur_task, gen, solver, ratio, device, TestDataLoaders, solver_acc_dict):
    """
    evaluate solver's accuracy
    
    """
    gen.eval()
    solver.eval()
    accuracy_list = []
    # solver_loss = 0.0
    celoss = torch.nn.CrossEntropyLoss().to(device) if torch.cuda.is_available() else torch.nn.CrossEntropyLoss()
    _TestDataLoaders = TestDataLoaders[:cur_task+1]

    for i, testdataloader in enumerate(_TestDataLoaders):
        total = 0
        correct = 0
        for data in testdataloader:
            x, y = data
            total += x.shape[0]
            if torch.cuda.is_available():
                x = x.to(device)
                y = y.to(device)

            with torch.autograd.no_grad():
                output = torch.max(solver(x), dim=1)[1]
                correct += (output == y).sum().item()

        accuracy = (correct * 100) / total
        accuracy_list.append(accuracy)
        solver_acc_dict[i][cur_task] = accuracy

    accuracy = np.average(np.array(accuracy_list))
    print("Task {} solver's accuracy(%): {}\n".format(cur_task+1, accuracy))
    return accuracy