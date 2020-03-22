import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image

import sys, os, shutil, random
import argparse

import collections

parser = argparse.ArgumentParser(description='PyTorch Shallow CNN')
parser.add_argument("--resume", type=str, default="", help="path of saved model")
parser.add_argument("--cmodel", type=str, default="", help="path of saved compare model")
args = parser.parse_args()

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class shallowCNN(nn.Module):
	def __init__(self):
		#self.config = config
		super(shallowCNN, self).__init__()
		# [in, out, kernel_size, stride, padding]
		self.bn0 = nn.BatchNorm2d(3)
		self.max_pool0 = nn.MaxPool2d(2, 2)

		self.conv1 = nn.Conv2d(3, 16, 5, 1, 2)
		self.bn1 = nn.BatchNorm2d(16)
		self.max_pool1 = nn.MaxPool2d(2, 2)

		self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
		self.bn2 = nn.BatchNorm2d(32)
		self.max_pool2 = nn.MaxPool2d(2, 2)

		self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
		self.bn3 = nn.BatchNorm2d(64)
		self.max_pool3 = nn.MaxPool2d(2, 2)

		self.conv4 = nn.Conv2d(64, 128, 5, 1, 2)
		self.bn4 = nn.BatchNorm2d(128)
		self.max_pool4 = nn.MaxPool2d(2, 2)

		self.conv5 = nn.Conv2d(128, 256, 5, 1, 2)
		self.bn5 = nn.BatchNorm2d(256)
		self.max_pool5 = nn.MaxPool2d(2, 2)
		self.linear1 = nn.Linear(256 * 4 * 4, 2048)
		self.linear2 = nn.Linear(2048, 346)

		#self.linear3 = nn.Linear(256 * 4 * 4, 1)
		#self.linear2 = nn.Linear(2048, 1)
		#self.linear1 = None
		#self.linear2 = None
	def forward(self, x, xx):
		#print(inputs)
		#print(len(inputs))
		x = self.max_pool0(self.bn0(x))
		x = self.max_pool1(F.leaky_relu(self.bn1(self.conv1(x))))
		x = self.max_pool2(F.leaky_relu(self.bn2(self.conv2(x))))
		x = self.max_pool3(F.leaky_relu(self.bn3(self.conv3(x))))
		x = self.max_pool4(F.leaky_relu(self.bn4(self.conv4(x))))
		x = self.max_pool5(F.leaky_relu(self.bn5(self.conv5(x))))
		#print(x.size(), x.size(1) * x.size(2) * x.size(3))
		#exit()
		x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

		xx = self.max_pool0(self.bn0(xx))
		xx = self.max_pool1(F.leaky_relu(self.bn1(self.conv1(xx))))
		xx = self.max_pool2(F.leaky_relu(self.bn2(self.conv2(xx))))
		xx = self.max_pool3(F.leaky_relu(self.bn3(self.conv3(xx))))
		xx = self.max_pool4(F.leaky_relu(self.bn4(self.conv4(xx))))
		xx = self.max_pool5(F.leaky_relu(self.bn5(self.conv5(xx))))
		xx = xx.view(-1, xx.size(1) * xx.size(2) * xx.size(3)) #[M, 4096]

		#xx

		xxx = F.sigmoid(torch.cat([x, xx], dim=1)) #[M, 4096 * 2]
		#xxx = F.dropout(xxx, p=0.5)
		xxx = F.sigmoid(self.linear3(xxx))
		#xxx = F.dropout(xxx, p=0.5)
		#xxx = F.sigmoid(xxx)
		return xxx

class inception_v3(nn.Module):
    """docstring for inception_v3"""
    def __init__(self):
        super(inception_v3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, 512)
        self.model.AuxLogits.fc = nn.Linear(768, 512)
        self.linear = nn.Linear(1024, 1)
    def forward(self, x, xx):
        #print(self.model(x)[0].size(), self.model(x)[1].size())
        if self.training:
        	x = self.model(x)[0] + self.model(x)[1]
       		xx = self.model(xx)[0] + self.model(xx)[1]
       	else:
       		#print(self.model(x).size())
       		x = self.model(x)
       		xx = self.model(xx)
        cosine_dis = (F.cosine_similarity(x, xx) + 1) / 2
        xxx = self.linear(torch.cat([x, xx], dim=1))
        xxx= F.sigmoid(F.dropout(xxx, p=0.5))
        return xxx, cosine_dis


train_info = pd.read_csv("processed_train_info.csv")

processed_train_info = train_info.drop(['title', 'style', 'genre', 'date', 'Unnamed: 6', 'Unnamed: 0'], axis=1)

filenames = processed_train_info['filename'].tolist()
artists = processed_train_info['artist'].tolist()

artist_dict = {}
for i in range(len(artists)):
    if artists[i] not in artist_dict:
        artist_dict[artists[i]] = [filenames[i]]
    else:
        artist_dict[artists[i]].append(filenames[i])

def pick_positive_pair(artists, artist_dict):
	artist = np.random.choice(artists)
	pair = np.random.choice(artist_dict[artist], 2)
	im1 = np.array(Image.open("resized_train/" + pair[0]))
	im2 = np.array(Image.open("resized_train/" + pair[1]))
	if len(im1.shape) == 2:
		pad_matrix = np.zeros((299, 299, 2))
		im1 = im1.reshape(299, 299, 1)
		im1 = np.concatenate((im1, pad_matrix), axis=2)
	if len(im2.shape) == 2:
		im2 = im2.reshape(299, 299, 1)
		pad_matrix = np.zeros((299, 299, 2))
		im2 = np.concatenate((im2, pad_matrix), axis=2)
	im1 = im1[:,:,:3].reshape(1, 299, 299, 3).transpose(0, 3, 1, 2)
	im2 = im2[:,:,:3].reshape(1, 299, 299, 3).transpose(0, 3, 1, 2)
	
	return [im1, im2]

def pick_negative_pair(artists, artist_dict):
	artist_pair = np.random.choice(artists, 2)
	pair = [np.random.choice(artist_dict[artist_pair[0]]), np.random.choice(artist_dict[artist_pair[1]])]
	im1 = np.array(Image.open("resized_train/" + pair[0]))
	im2 = np.array(Image.open("resized_train/" + pair[1]))
	if len(im1.shape) == 2:
		im1 = im1.reshape(299, 299, 1)
		pad_matrix = np.zeros((299, 299, 2))
		im1 = np.concatenate((im1, pad_matrix), axis=2)
	im2 = np.array(Image.open("resized_train/" + pair[1]))
	if len(im2.shape) == 2:
		im2 = im2.reshape(299, 299, 1)
		pad_matrix = np.zeros((299, 299, 2))
		im2 = np.concatenate((im2, pad_matrix), axis=2)
	im1 = im1[:,:,:3]
	im1 = im1.reshape(1, 299, 299, 3).transpose(0, 3, 1, 2)
	im2 = im2[:,:,:3]
	im2 = im2.reshape(1, 299, 299, 3).transpose(0, 3, 1, 2)
	return [im1, im2]

def load_pretrained_model(model):
	print("=> loading checkpoint '{}'".format(args.resume))
	checkpoint = torch.load(args.resume)
	#START_EPOCH = checkpoint["epoch"] + 1
	#model.load_state_dict(checkpoint['state_dict'])
	print("=> loaded checkpoint '{}' (epoch {})"
		.format(args.resume, checkpoint['epoch']))
	return model

def load_compare_model(model):
	print("=> loading compare_model '{}'".format(args.cmodel))
	checkpoint = torch.load(args.cmodel)
	step = checkpoint["step"] + 1
	model.load_state_dict(checkpoint['state_dict'])
	print("=> loaded checkpoint '{}' (step {})"
		.format(args.cmodel, checkpoint['step']))
	return model, step

LR = 1e-5
STEPS = int(250)

#model = shallowCNN()


if args.resume:
	print("Loading pretrained model...")
	model = load_pretrained_model(model)
	model.linear3 = nn.Linear(256 * 4 * 4 * 2, 1)
	model.bn6 = nn.BatchNorm1d(8192)
	
model = inception_v3()
START_STEP = 0
if args.cmodel:
	model, START_STEP = load_compare_model(model)

'''
optim_params = []
for p in model.named_parameters():
	if p[0].find("linear3") != -1:
		p[1].requires_grad = True
		optim_params.append(p[1])
	else:
		p[1].requires_grad = False
'''
model.cuda()
loss_fn = nn.BCELoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-3)


'''
for p in model.named_parameters():
	if p[0].find("linear") != -1:
		p[1].requires_grad = True
		optim_params.append(p[1])
	else:
		p[1].requires_grad = False
'''
model.train()
loss_list = collections.deque()

random_crop = transforms.Compose([
	transforms.RandomCrop(224, 224)
])
for i in range(START_STEP, STEPS):
	pos_pair_examples = []
	for _ in range(64):
		pos_pair_examples.append(pick_positive_pair(artists, artist_dict))
	neg_pair_examples = []
	for _ in range(64):
		neg_pair_examples.append(pick_negative_pair(artists, artist_dict))
		
	for pos_pair in pos_pair_examples:
		pos_pair[0] = Variable(torch.from_numpy(pos_pair[0]).float(), requires_grad=False).cuda()
		pos_pair[1] = Variable(torch.from_numpy(pos_pair[1]).float(), requires_grad=False).cuda()
		
		ones = Variable(torch.ones(1, 1).float(), requires_grad=False).cuda()
		zeros = Variable(torch.zeros(1, 1).float(), requires_grad=False).cuda()
		scores, cosine_dis = model(pos_pair[0], pos_pair[1])

		model.zero_grad()
		score_loss = loss_fn(scores, ones)
		cosine_loss = mse_loss(cosine_dis, ones)
		loss = score_loss
		#loss = score_loss + cosine_loss
		loss_list.append(loss.data[0])
		if len(loss_list) > 1000:
			loss_list.popleft()
		loss.backward()
		optimizer.step()
		sys.stdout.write("Step: %d, training: P, total loss: %.4f, score loss: %.4f, cosine loss:%.4f\r" % 
			(i, np.mean(loss_list), score_loss.data[0], cosine_loss.data[0]))

	for neg_pair in neg_pair_examples:
		neg_pair[0] = Variable(torch.from_numpy(neg_pair[0]).float(), requires_grad=False).cuda()
		neg_pair[1] = Variable(torch.from_numpy(neg_pair[1]).float(), requires_grad=False).cuda()
		
		ones = Variable(torch.ones(1, 1).float(), requires_grad=False).cuda()
		zeros = Variable(torch.zeros(1, 1).float(), requires_grad=False).cuda()
		scores, cosine_dis = model(neg_pair[0], neg_pair[1])

		model.zero_grad()
		score_loss = loss_fn(scores, zeros)
		cosine_loss = mse_loss(cosine_dis, zeros)
		loss = score_loss
		#loss = score_loss + cosine_loss
		loss_list.append(loss.data[0])
		if len(loss_list) > 1000:
			loss_list.popleft()
		loss.backward()
		optimizer.step()
		sys.stdout.write("Step: %d, training: N, total loss: %.4f, score loss: %.4f, cosine loss:%.4f\r" % 
			(i, np.mean(loss_list), score_loss.data[0], cosine_loss.data[0]))
	if (i + 1) % 5 == 0:
		print()
		print("Saving models...")
		save_checkpoint({
	            'step': i + 1,
	            'state_dict': model.state_dict(),
	            'optimizer' : optimizer.state_dict(),
	            'loss': loss
	        }, False, filename="compare_model.tar")
		print("Model saved!")
print()
# make submission
print("predict the result")
res_dict = {}
model.eval()
submission_info = pd.read_csv("submission_info.csv", index_col=['index'])
for i in range(len(submission_info["img1"])):
	im1 = np.array(Image.open("resized_test/" + submission_info["img1"][i]))
	if len(im1.shape) == 2:
		tmp_pad = np.zeros((299, 299, 2))
		im1 = im1.reshape(299, 299, 1)
		im1 = np.concatenate([im1, tmp_pad], axis=2)
	im1 = im1[:,:,:3]
	im1 = im1.reshape(1, 299, 299, 3).transpose(0, 3, 1, 2)

	im2 = np.array(Image.open("resized_test/" + submission_info["img2"][i]))
	if len(im2.shape) == 2:
		tmp_pad = np.zeros((299, 299, 2))
		im2 = im2.reshape(299, 299, 1)
		im2 = np.concatenate([im2, tmp_pad], axis=2)
	im2 = im2[:,:,:3]
	im2 = im2.reshape(1, 299, 299, 3).transpose(0, 3, 1, 2)
	im1 = Variable(torch.from_numpy(im1).float(), requires_grad=False).cuda()
	im2 = Variable(torch.from_numpy(im2).float(), requires_grad=False).cuda()
	#print(im1.size())
	scores, cosine_dis = model(im1, im2)
	res_dict[i] = scores.data[0][0]

	if i % 10 == 0:
		sys.stdout.write("predicting: %d/%d\r"%(i, len(submission_info["img1"])))

print("Creating submission file...")
output_series = pd.Series(res_dict)
output_series.to_csv("submission.csv", index_label="index", header=["sameArtist"])
print("submission.csv created!")
