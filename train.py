import torch
import torchaudio
from dataset.dataloader import padding_tensor
import numpy as np
from models.gnn import GCNN
from torch.functional import F


emo ={'ang':0, 'hap':1, 'neu':2, 'sad':3}


def train(model, device, trainloader, max_len, num_epochs):
	model.to(device)
	model.train()

	#criterion = torch.nn.BCEWithLogitsLoss()
	criterion = torch.nn.CrossEntropyLoss()
	epochs_stop = 5
	min_loss = np.Inf
	epoch_min_loss = np.Inf
	no_improve = 0
	total_step = len(trainloader)
	loss_list = []
	acc_list = []

	#Graph model

	g_model = GCNN()
	g_model.train()

	optimizer = torch.optim.Adam([
		dict(params=model.parameters(), weight_decay=5e-4),
		dict(params=g_model.gconv1.parameters(), weight_decay=5e-4),
		dict(params=g_model.gconv2.parameters(), weight_decay=0)
	], lr=0.01)

	for epoch in range(1, num_epochs+1):
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			try:
				audio = [torchaudio.load(file)[0] for a in data.y for file in a]
			except:
				continue
			audio = padding_tensor(audio, max_len)
			labels = [emo[d[d.find('/')+1:d.find('/')+4]] for a in data.y for d in a]
			labels=torch.as_tensor(labels)
			audio.to(device)
			labels.to(device)
			x_embedding = model.acoustic_model(audio)
			data.x = x_embedding
			out = g_model(data)
			loss1 = criterion(out, labels)
			loss = F.nll_loss(out, labels)

			loss_list.append(loss.item())
			if loss < min_loss:
				min_loss = loss
			# Backprop and perform Adam optimization
			loss.backward()
			optimizer.step()

			# Track the accuracy
			total = labels.size(0)
			_, predicted = torch.max(out.data, 1)
			print(predicted)
			print(labels)
			correct = (predicted == labels).sum().item()
			acc_list.append(correct / total)
			print(acc_list)

			if (i + 1) % 2:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
				      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
				              (correct / total) * 100))
			if min_loss < epoch_min_loss:
				epoch_min_loss = min_loss
				no_improve = 0
			else:
				no_improve += 1
			if no_improve == epochs_stop:
				break
			else:
				continue
			loss_list.append('-----' + str(epoch) + '-----')
			write_file('accuracy.txt', acc_list)
			write_file('loss.txt', loss_list)

'''
##Test Model###
	model.eval()
	correct = 0
	total = 0
	y = []
	y_predicted = []
	with torch.no_grad():
		for labels, sounds in testloader:
			if torch.cuda.is_available():
				labels = labels.cuda()
				sounds = sounds.cuda()
			outputs = model(sounds)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			y_predicted.append(outputs.numpy())
			y.append(labels.numpy())

	print('Accuracy: %d %%' % (100 * correct / total))
	write_file('accuracy_test.txt', [(100 * correct / total)])


'''

