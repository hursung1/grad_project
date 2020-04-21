import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import pyfiles.models as models
import pyfiles.lib as lib

tasks = 12
epochs = 100
learning_rate = 1e-3
batch_size = 64
data_path = './data/RADAR'

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
train_data, train_labels, test_data, test_labels = lib.RADARLoader(data_path, tasks, category, device)

net = models.ConvolutionNetwork().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

if not os.path.isdir("./log"):
    os.mkdir("./log")

logfile_name = "./log/logfile_training_%d_%d_%d_%d_%d.txt" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
log_file = open(logfile_name, "w")

running_loss = 0
avg_acc = {}

net.train()
for task in range(tasks):
    msg = "Task %d\n"%(task+1)
    print(msg)
    log_file.write(msg)
    
    for epoch in range(epochs):
    #for key, _ in category.items():
        _train_data = train_data[task].view(-1, 1, 128, 128)
        _train_label = train_labels.view(-1)
        
#         print(_train_data.shape)
#         print(_train_label.shape)
        optimizer.zero_grad()
        net.zero_grad()
        outputs = net(_train_data)
        loss = criterion(outputs, _train_label)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        if epoch % 10 == 9:
            msg = 'Epoch: %d, AVG. loss: %.3f\n'% (epoch + 1, running_loss)
            print(msg)
            log_file.write(msg)
            running_loss = 0

    total = 0
    correct = 0
    net.eval()
    test_acc = []
    for test_task in range(task+1):
        _test_data = test_data[test_task].view(-1, 1, 128, 128)
        _test_label = test_labels.view(-1)#.unsqueeze(0)
#         tmp = _test_label.clone()
#         for _ in range(task):
#             _test_label = torch.cat((_test_label, tmp))

#         _test_label = _test_label.view(-1)
#         print(_test_label.shape)

        output = net(_test_data)
        _, predicted = torch.max(output.data, dim=1)
        total += _test_label.shape[0]
        correct += (predicted == _test_label).sum()
    
        acc = correct.cpu().numpy()*100/total
        test_acc.append(acc)
        
    print(test_acc)
    acc = torch.tensor(test_acc).mean()
    msg = 'Average accuracy %d %%\n' % (acc)
    print(msg)
    log_file.write(msg)
    avg_acc[task+1] = acc

log_file.close()

plt.plot(avg_acc.keys(), avg_acc.values())
plt.xlabel('# of tasks')
plt.ylabel('accuracy')
plt.savefig('result.png', dpi=200)