import torch
from dataset.dataloader import padding_tensor, audio_loader, write_file
import numpy as np
import os
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset


def pretrain(model, device, num_epochs, batch_size, train_path, test_path):
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	scheduler = ExponentialLR(optimizer, gamma=0.9)
	criterion = torch.nn.CrossEntropyLoss()
	epochs_stop = 3
	epoch_min_loss = None
	no_improve = 0
	acc_list = []
	start_epoch=1
	checkpoint_path = 'speech_representation.pt'

	#Check if a checkpoint exists to continue training
	if os.path.isfile('pretrained/speech_representation.pt'):
		checkpoint = torch.load('pretrained/speech_representation.pt')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		loss = checkpoint['loss']

	train_data, train_labels, max_len = audio_loader(train_path)
	test_data, test_labels, max_len2 = audio_loader(test_path)
	max_len=max(max_len, max_len2)
	train_data = padding_tensor(train_data, max_len)
	train_data = TensorDataset(train_data, train_labels)
	test_data = padding_tensor(test_data, max_len)
	test_data = TensorDataset(test_data, test_labels)
	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
	total_step = len(train_dataloader)

	model.train()

	for epoch in range(start_epoch, num_epochs):
		epoch_loss=[]

		for i, data in enumerate(train_dataloader, 0):
			optimizer.zero_grad()
			audio =data[0]
			labels=data[1]
			audio.to(device)
			labels.to(device)
			out = model(audio)
			loss = criterion(out, labels)

			# Track the accuracy
			total = labels.size(0)
			_, predicted = torch.max(out.data, 1)
			correct = (predicted == labels).sum().item()
			acc_list.append(correct / total)

			# Backprop and perform Adam optimization
			loss.backward()
			optimizer.step()
			print(loss)

			### Add loss per iteration to check later in epoch ###
			epoch_loss.append(loss)

			# Scheduler decay
			scheduler.step()

			if (i + 1) % 10:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
				      .format(epoch , num_epochs, i + 1, total_step, loss.item(),
				              (correct / total) * 100))

			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss,
			}, checkpoint_path)

			torch.save(model.state_dict(), 'speech_model.pt')


		### Epoch check ###
		e_loss = sum(epoch_loss)/len(epoch_loss)
		if epoch_min_loss == None:
			epoch_min_loss = e_loss
		elif e_loss < epoch_min_loss:
			epoch_min_loss = e_loss
			no_improve = 0
		else:
			no_improve += 1
		if no_improve == epochs_stop:
			break


##Test Model###
	model.eval()
	correct = 0
	total = 0
	y = []
	y_predicted = []
	with torch.no_grad():
		for data in test_dataloader:
			audio = data[0]
			labels = data[1]
			outputs = model(audio)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			y_predicted.append(outputs.numpy())
			y.append(labels.numpy())

	print('Accuracy: %d %%' % (100 * correct / total))
	write_file('accuracy_test.txt', [(100 * correct / total)])