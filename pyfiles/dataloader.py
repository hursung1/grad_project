import torch
import torchvision
import numpy as np

class PermutedMNISTDataLoader(torchvision.datasets.MNIST):
    
    def __init__(self, source='./mnist_data', train = True, shuffle_seed = None):
        super(PermutedMNISTDataLoader, self).__init__(source, train, download=True)
        
        self.train = train
        if self.train:
            self.permuted_train_data = torch.stack(
                [img.type(dtype=torch.float32).view(-1)[shuffle_seed] / 255.0
                    for img in self.train_data])
        else:
            self.permuted_test_data = torch.stack(
                [img.type(dtype=torch.float32).view(-1)[shuffle_seed] / 255.0
                    for img in self.test_data])
            
    def __getitem__(self, index):
        
        if self.train:
            input, label = self.permuted_train_data[index], self.train_labels[index]
        else:
            input, label = self.permuted_test_data[index], self.test_labels[index]
        
        return input, label

    def sample(self, size):
        return [img for img in self.permuted_train_data[random.sample(range(len(self), size))]]
    
    '''
    def __len__(self):
        if self.train:
            return self.train_data.size()
        else:
            return self.test_data.size()
    '''
    
    
class RADARDataSet(torch.utils.data.Dataset):
    def __init__(self, source='./RADAR', train=True, transform=None):
        self.train = train
        self.transform = transform
        
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass