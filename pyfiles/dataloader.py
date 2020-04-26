import torch
import torchvision
import numpy as np

class MNIST_PermutedDataset(torchvision.datasets.MNIST):
    """
    Permuted MNIST Dataset for Domain Learning
    """
    def __init__(self, source='./mnist_data', train = True, transform=None, shuffle_seed = None, download=True):
        super(MNIST_PermutedDataset, self).__init__(source, train=train, transform=transform, download=True)
        
        self.train = train
        self.num_data = 0

        if shuffle_seed is None:
            shuffle_seed = np.arange(28*28)
            np.random.shuffle(shuffle_seed)
        
        if self.train:
            self.permuted_train_data = []
            self.permuted_train_labels = []
            for i, data in enumerate(self.train_data):
                data_shape = (1, *data.shape)
                _data = transform(data.numpy()) if transform is not None else data
                _data = _data.reshape(-1)[shuffle_seed].reshape(data_shape)
                self.permuted_train_data.append(_data)
            
        else:
            self.permuted_test_data = []
            self.permuted_test_labels = []
            for i, data in enumerate(self.test_data):
                data_shape = (1, *data.shape)
                _data = transform(data.numpy()) if transform is not None else data
                _data = _data.reshape(-1)[shuffle_seed].reshape(data_shape)
                self.permuted_test_data.append(_data)
            
    def __getitem__(self, index):
        
        if self.train:
            input, label = self.permuted_train_data[index], self.train_labels[index]
        else:
            input, label = self.permuted_test_data[index], self.test_labels[index]
        
        return input, label

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
    
    def getNumData(self):
        return self.num_data
    

class MNIST_IncrementalDataset(torchvision.datasets.MNIST):
    """
    MNIST Dataset for Incremental Learning
    """
    def __init__(self, 
                 source='./mnist_data', 
                 train=True,
                 transform=None,
                 download=False,
                 classes=range(10)):
        
        super(MNIST_IncrementalDataset, self).__init__(source, 
                                                       train, 
                                                       transform, 
                                                       download=True)
        self.train = train
        self.transform = transform

        if train:
            train_data = []
            train_labels = []
            for i in range(len(self.train_data)):
                if self.train_labels[i] in classes:
                    _data = transform(self.train_data[i].numpy()) if transform is not None else self.train_data[i]
                    # train_data.append(self.train_data[i].type(dtype=torch.float32))
                    train_data.append(_data)
                    train_labels.append(self.train_labels[i])
            
            self.TrainData = train_data
            self.TrainLabels = train_labels

        else:
            test_data = []
            test_labels = []
            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    _data = transform(self.test_data[i].numpy()) if transform is not None else self.test_data[i]
                    # test_data.append(self.test_data[i].type(dtype=torch.float32))
                    test_data.append(_data)
                    test_labels.append(self.test_labels[i])
            
            self.TestData = test_data
            self.TestLabels = test_labels

    def __getitem__(self, index):
        if self.train:
            return self.TrainData[index], self.TrainLabels[index]
        else:
            return self.TestData[index], self.TestLabels[index]

    def __len__(self):
        if self.train:
            return len(self.TrainLabels)
        else:
            return len(self.TestLabels)
        
        
class RADARDataSet(torch.utils.data.Dataset):
    def __init__(self, source='./RADAR', train=True, transform=None):
        self.train = train
        self.transform = transform
        
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass