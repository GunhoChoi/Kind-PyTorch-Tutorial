import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# Preprocessing 
# alphabet(0-25), space(26), start(27), end(28) -> 29 chars (0-28)

string = "hello pytorch"
chars = "abcdefghijklmnopqrstuvwxyz 01"
char_list = [i for i in chars]
batch_size = len(char_list)


def string_to_onehot(string):
	start = np.zeros(shape=len(char_list) ,dtype=int)
	end = np.zeros(shape=len(char_list) ,dtype=int)
	start[-2] = 1
	end[-1] = 1
	for i in string:
		idx = char_list.index(i)
		zero = np.zeros(shape=batch_size ,dtype=int)
		zero[idx]=1
		start = np.vstack([start,zero])
	output = np.vstack([start,end])
	return output


def onehot_to_word(onehot_1):
	onehot = torch.Tensor.numpy(onehot_1)
	return char_list[onehot.argmax()]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


# Hyperparameters

n_letters = len(char_list)
n_hidden = 100
#n_categories = len(char_list)
lr = 0.01
epochs = 100

rnn = RNN(n_letters, n_hidden, n_letters)

# Loss function & Optimizer

one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

# train

for i in range(epochs):
	rnn.zero_grad()
	total_loss = 0
	hidden = rnn.init_hidden()

	for j in range(one_hot.size()[0]-1):
		input = Variable(one_hot[j:j+1,:])
		output, hidden = rnn.forward(input, hidden)
		target = Variable(one_hot[j+1])
		loss = loss_func(output.view(-1),target.view(-1))
		total_loss += loss
		input = output

	total_loss.backward()
	optimizer.step()

	if i % 10 == 0:
		print(total_loss)

# test 

hidden = rnn.init_hidden()
input = Variable(one_hot[0:1,:])

for i in range(len(string)):
	output, hidden = rnn.forward(input, hidden)
	print(onehot_to_word(output.data))
	input = output
