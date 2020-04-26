import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch

import pyfiles.models as models
import pyfiles.lib as lib

tasks = 12
epochs = 100
learning_rate = 1e-3
batch_size = 64
data_path = '../data/RADAR'

category={0: 'boxingmoving', 
          1: 'boxingstill', 
          2: 'crawling', 
          3: 'running', 
          4: 'still', 
          5: 'walking', 
          6: 'walkinglow'}

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
dt = datetime.datetime.now()

### Data: (# of subject, # of classes, # of data, width, height)
train_data, train_labels, test_data, test_labels = lib.RADARLoader(data_path, category, device, "")

net = models.ConvolutionNetwork().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

logfile_name = "logfile_training_%d_%d_%d_%d_%d.txt" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
log_file = open(logfile_name, "w")

running_loss = 0
avg_acc = {}

net.train()
for epoch in range(epochs):
    for key, _ in category.items():
        _train_data = train_data[key].view(9*12, 1, 128, 128)
        _train_label = train_labels[key]
        #print(_train_data.shape)
        optimizer.zero_grad()
        outputs = net(_train_data)
        loss = criterion(outputs, _train_label)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

    if epoch % 5 == 4:
        msg = 'Epoch: %d, AVG. loss: %.3f\n'% (epoch + 1, running_loss)
        print(msg)
        log_file.write(msg)
        running_loss = 0

total = 0
correct = 0
net.eval()
for key, _ in category.items():
    _test_data = test_data[key].view(3*12, 1, 128, 128)
    output = net(_test_data)
    _, predicted = torch.max(output.data, dim=1)
    total += test_labels[key].shape[0]
    correct += (predicted == test_labels[key]).sum()
        
acc = correct.cpu().numpy()*100/total
msg = 'Average accuracy %d %%\n' % (acc)
print(msg)
log_file.write(msg)
log_file.close()