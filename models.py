# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image

import sys, os, shutil, random


# In[2]:


'''processed_train_info = pd.read_csv("processed_train_info.csv")
processed_train_info = processed_train_info.drop(['title', 'style', 'genre', 'date', 'Unnamed: 6', 'Unnamed: 0'], axis=1)
filenames = processed_train_info['filename'].tolist()
artists = processed_train_info['artist'].tolist()'''


# In[3]:


'''input_images = np.random.randint(low=0, high=255,size=(12747, 299, 299, 3), dtype=np.uint8)
for i, filename in enumerate(filenames):
    if i  % 100 == 0:
        sys.stdout.write("%d\r"%i)
    im = np.asarray(Image.open("resized_train/"+filename), dtype=np.uint8)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
        new_zeros = np.zeros(shape=(299,299,2), dtype='uint8')
        im = np.concatenate((im, new_zeros), axis=2)
        print(im.shape)
    input_images[i] = im[:, :, :3]'''


# In[4]:


#np.save("input_images.npz", input_images)


# In[5]:


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# In[6]:


artists = np.load("labels.npy")
input_images = np.load("input_images.npz.npy")


# In[7]:


input_images = input_images.transpose(0, 3, 1, 2)
#input_images = input_images


# In[8]:


x_train, x_valid, y_train, y_valid = train_test_split(input_images, artists, test_size=0.2)


# In[9]:


class shallowCNN(nn.Module):
    def __init__(self):
        #self.config = config
        super(shallowCNN, self).__init__()
        # [in, out, kernel_size, stride, padding]
        self.bn0 = nn.BatchNorm2d(3)
        self.max_pool0 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.max_pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.max_pool4 = nn.MaxPool2d(2, 2)
        
        self.linear1 = nn.Linear(128 * 9 * 9, 2048)
        self.linear2 = nn.Linear(2048, 346)
    def forward(self, x):
        x = self.max_pool0(self.bn0(x))
        x = self.max_pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.max_pool2(F.elu(self.bn2(self.conv2(x))))
        x = self.max_pool3(F.elu(self.bn3(self.conv3(x))))
        x = self.max_pool4(F.elu(self.bn4(self.conv4(x))))
        #print(x.size())
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.elu(self.linear1(x))
        x = F.log_softmax(self.linear2(x))
        return x


# In[10]:


model = shallowCNN()
model.cuda()


# In[11]:


EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 4
best_prec = 0
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.NLLLoss()
loss_list = []


# In[12]:


for epoch in range(EPOCHS):
    model.train()
    index = np.random.permutation(len(x_train))
    x_train = x_train[index]
    y_train = y_train[index]
    for i in range(len(x_train) // BATCH_SIZE):
        model.zero_grad()
        
        start_ix = i * BATCH_SIZE
        end_ix = (i + 1) * BATCH_SIZE
        x_batch = x_train[start_ix:end_ix]
        y_batch = y_train[start_ix:end_ix]
        x_batch = Variable(torch.from_numpy(x_batch).float(), requires_grad=False).cuda()
        y_batch = Variable(torch.LongTensor(y_batch), requires_grad=False).cuda()
        
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss_list.append(loss.data[0])
        if len(loss_list) > 10000:
            loss_list = loss_list[1:]
        loss.backward()
        optimizer.step()
        if i % 100:
            sys.stdout.write(" "*80 + "\r")
            sys.stdout.write("Epoch: %d, Step %d/%d, loss: %.4f\r" % 
                             (epoch, i, len(x_train) // BATCH_SIZE + 1, np.mean(loss_list)))
            
    model.zero_grad()
    model.eval()
    correct = 0
    for i in range(len(x_valid) // BATCH_SIZE):
        start_ix = i * BATCH_SIZE
        end_ix = (i + 1) * BATCH_SIZE
        x_batch = x_valid[start_ix:end_ix]
        y_batch = y_valid[start_ix:end_ix]
        x_batch = Variable(torch.from_numpy(x_batch).float(), requires_grad=False).cuda()
        y_batch = Variable(torch.LongTensor(y_batch), requires_grad=False).cuda()
        logits = model(x_batch)
        loss += loss_fn(logits, y_batch)
        pred = logits.data.max(1, keepdim=True)[1]
        correct += pred.eq(y_batch.data.view_as(pred)).cpu().sum()
    
    correct /= len(x_valid)
    loss /= len(x_valid)
    print("Epoch: %d, precision: %.4f, loss: %.4f" % (epoch, correct, loss.data[0]))
    if correct > best_prec:
        best_prec = correct
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_prec': best_prec
        }, best_prec == correct)
