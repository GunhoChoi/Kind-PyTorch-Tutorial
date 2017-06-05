import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.utils as v_utils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Set Hyperparameters

epoch = 1000
batch_size = 6000
learning_rate = 0.0002
num_gpus = 4

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)

class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.layer1 = nn.Linear(100,7*7*256)
		self.layer2 = nn.Sequential(OrderedDict([
				('conv1', nn.ConvTranspose2d(256,128,3,2,1,1)),
				('relu1', nn.LeakyReLU()),
				('bn1', nn.BatchNorm2d(128)),
				('conv2', nn.ConvTranspose2d(128,64,3,1,1)),
				('relu2', nn.LeakyReLU()),
				('bn2', nn.BatchNorm2d(64))
			]))
		self.layer3 = nn.Sequential(OrderedDict([
				('conv3',nn.ConvTranspose2d(64,16,3,1,1)),
				('relu3',nn.LeakyReLU()),
				('bn3',nn.BatchNorm2d(16)),
				('conv4',nn.ConvTranspose2d(16,1,3,2,1,1)),
				('relu4',nn.LeakyReLU())
			]))

	def forward(self,z):
		out = self.layer1(z)
		out = out.view(batch_size//num_gpus,256,7,7)
		out = self.layer2(out)
		out = self.layer3(out)
		return out


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.layer1 = nn.Sequential(OrderedDict([
				('conv1',nn.Conv2d(1,16,3,padding=1)),   # batch x 16 x 28 x 28
				('relu1',nn.LeakyReLU()),
				('bn1',nn.BatchNorm2d(16)),
				('conv2',nn.Conv2d(16,32,3,padding=1)),  # batch x 32 x 28 x 28
				('relu2',nn.LeakyReLU()),
				('bn2',nn.BatchNorm2d(32)),
				('max1',nn.MaxPool2d(2,2))   # batch x 32 x 14 x 14
			]))
		self.layer2 = nn.Sequential(OrderedDict([
				('conv3',nn.Conv2d(32,64,3,padding=1)),  # batch x 64 x 14 x 14
				('relu3',nn.LeakyReLU()),
				('bn3',nn.BatchNorm2d(64)),
				('max2',nn.MaxPool2d(2,2)),
				('conv4',nn.Conv2d(64,128,3,padding=1)),  # batch x 128 x 7 x 7
				('relu4',nn.LeakyReLU())
			]))
		self.fc = nn.Sequential(
				nn.Linear(128*7*7,1),
				nn.Sigmoid()
			)

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(batch_size//num_gpus, -1)
		out = self.fc(out)
		return out
    
generator = nn.DataParallel(Generator()).cuda()
discriminator = nn.DataParallel(Discriminator()).cuda()

loss_func = nn.BCELoss()
gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate)
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

ones_label = Variable(torch.ones(batch_size,1)).cuda()
zeros_label = Variable(torch.zeros(batch_size,1)).cuda()

# parameter initialization

gen_params = generator.state_dict().keys()
dis_params = discriminator.state_dict().keys()

# model restore

try:
	generator, discriminator = torch.load('./model/vanilla_gan.pkl')
	print("\n--------model restored--------\n")
except:
	print("\n--------model not restored--------\n")
	pass

# train

for i in range(epoch):
	for j,(image,label) in enumerate(train_loader):
		image = Variable(image).cuda()
		
		for k in range(5):
			z = Variable(torch.rand(batch_size,100)).cuda()
			gen_optim.zero_grad()
			gen_fake = generator.forward(z)
			dis_fake = discriminator.forward(gen_fake)
			gen_loss = torch.sum(loss_func(dis_fake,ones_label))
			gen_loss.backward(retain_variables=True)
			gen_optim.step()

		dis_optim.zero_grad()
		dis_real = discriminator.forward(image)
		dis_loss = torch.sum(loss_func(dis_fake,zeros_label))+torch.sum(loss_func(dis_real,ones_label))
		dis_loss.backward()
		dis_optim.step()
		
	if i % 5 == 0:
		torch.save([generator,discriminator],'./model/vanilla_gan.pkl')
	
	print("{}th iteration gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))
	v_utils.save_image(gen_fake.data[0:20],"./gan_result/gen_{}.png".format(i), nrow=5)
