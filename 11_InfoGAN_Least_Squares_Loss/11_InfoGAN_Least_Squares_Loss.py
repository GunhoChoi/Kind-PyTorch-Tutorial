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
batch_size = 2000
learning_rate = 0.001
num_gpus = 2
discrete_latent_size = 10

# Download Data & Set Data Loader(input pipeline)

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)


def int_to_onehot(z_label):
    one_hot_array = np.zeros(shape=[len(z_label), discrete_latent_size])
    one_hot_array[np.arange(len(z_label)), z_label] = 1
    return one_hot_array


class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.layer1 = nn.Linear(110,7*7*256)				# [batch,110] -> [batch,7*7*256]		
		self.layer2 = nn.Sequential(OrderedDict([
				('conv1', nn.ConvTranspose2d(256,128,3,2,1,1)),	# [batch,256,7,7] -> [batch,128,14,14]
				('relu1', nn.LeakyReLU()),
				('bn1', nn.BatchNorm2d(128)),
				('conv2', nn.ConvTranspose2d(128,64,3,1,1)),	# [batch,128,14,14] -> [batch,64,14,14]
				('relu2', nn.LeakyReLU()),
				('bn2', nn.BatchNorm2d(64))
			]))
		self.layer3 = nn.Sequential(OrderedDict([
				('conv3',nn.ConvTranspose2d(64,16,3,1,1)),	# [batch,64,14,14] -> [batch,16,14,14]
				('relu3',nn.LeakyReLU()),
				('bn3',nn.BatchNorm2d(16)),
				('conv4',nn.ConvTranspose2d(16,1,3,2,1,1)),	# [batch,16,14,14] -> [batch,1,28,28]
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
				('conv1',nn.Conv2d(1,16,3,padding=1)),   # [batch,1,28,28] -> [batch,16,28,28]
				('relu1',nn.LeakyReLU()),
				('bn1',nn.BatchNorm2d(16)),
				('conv2',nn.Conv2d(16,32,3,padding=1)),  # [batch,16,28,28] -> [batch,32,28,28]
				('relu2',nn.LeakyReLU()),
				('bn2',nn.BatchNorm2d(32)),
				('max1',nn.MaxPool2d(2,2))   		 # [batch,32,28,28] -> [batch,32,14,14]
			]))
		self.layer2 = nn.Sequential(OrderedDict([
				('conv3',nn.Conv2d(32,64,3,padding=1)),  # [batch,32,14,14] -> [batch,64,14,14] 
				('relu3',nn.LeakyReLU()),
				('bn3',nn.BatchNorm2d(64)),	
				('max2',nn.MaxPool2d(2,2)),		 # [batch,64,14,14] -> [batch,64,7,7]
				('conv4',nn.Conv2d(64,128,3,padding=1)), # [batch,64,7,7] -> [batch,128,7,7]
				('relu4',nn.LeakyReLU())
			]))
		self.fc = nn.Sequential(
				nn.Linear(128*7*7,1),
				nn.Sigmoid()
			)
		self.fc2 = nn.Sequential(
					nn.Linear(128*7*7,10),
					nn.LeakyReLU(),
			)

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.view(batch_size//num_gpus, -1)
		output = self.fc(out)
		label = self.fc2(out)
		return output,label
    
# put class instance on multi gpu

generator = nn.DataParallel(Generator(),device_ids=[0,1]).cuda()
discriminator = nn.DataParallel(Discriminator(),device_ids=[0,1]).cuda()

# put labels on multi gpu

ones_label = Variable(torch.ones(batch_size,1)).cuda()
zeros_label = Variable(torch.zeros(batch_size,1)).cuda()

# loss function and optimizer 
# this time, use LSGAN loss(https://arxiv.org/abs/1611.04076v2)

loss_func = nn.MSELoss()
gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate)
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# model restore

try:
	generator, discriminator = torch.load('./model/infogan.pkl')
	print("\n--------model restored--------\n")
except:
	print("\n--------model not restored--------\n")
	pass

# train 

for i in range(epoch):
    for j,(image,label) in enumerate(train_loader):
        
        # put image & label on gpu
        image = Variable(image).cuda()
        label = torch.from_numpy(int_to_onehot(label.numpy()))
        label = Variable(label.type_as(torch.FloatTensor())).cuda()
    
        # generator 
        for k in range(2):
            z_random = np.random.rand(batch_size,100)
            z_label = np.random.randint(0, 10, size=batch_size)
            
            # change first 10 labels from random to 0~9          
            for l in range(10):
                z_label[l]=l
            
            # preprocess z
            z_label_onehot = int_to_onehot(z_label)
            z_concat = np.concatenate([z_random, z_label_onehot], axis=1)
            z = Variable(torch.from_numpy(z_concat).type_as(torch.FloatTensor())).cuda()
            z_label_onehot = Variable(torch.from_numpy(z_label_onehot).type_as(torch.FloatTensor())).cuda()

            # calculate loss and apply gradients
            # gen_loss = gan loss(fake) + categorical loss
            gen_optim.zero_grad()
            gen_fake = generator.forward(z)
            dis_fake,label_fake = discriminator.forward(gen_fake)
            gen_loss = torch.sum(loss_func(dis_fake,ones_label)) + discrete_latent_size * torch.sum(loss_func(label_fake,z_label_onehot))
            gen_loss.backward(retain_variables=True)
            gen_optim.step()

        # discriminator
        # dis_loss = gan_loss(fake & real) + categorical loss
        dis_optim.zero_grad()
        dis_real, label_real = discriminator.forward(image)
        dis_loss = torch.sum(loss_func(dis_fake,zeros_label))+torch.sum(loss_func(dis_real,ones_label)) + discrete_latent_size * torch.sum(loss_func(label_real,label))
        dis_loss.backward()
        dis_optim.step()
    
    # model save
    if i % 5 == 0:
        torch.save([generator,discriminator],'./model/infogan.pkl')

    # print loss and image save
    print("{}th iteration gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))
    v_utils.save_image(gen_fake.data[0:20],"./result/gen_{}.png".format(i), nrow=5)
