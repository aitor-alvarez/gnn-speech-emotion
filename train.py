import torch
import torchaudio
from dataset.dataloader import padding_tensor, get_subgraph, write_file
import numpy as np
import os


emo ={'ang':0, 'hap':1, 'neu':2, 'sad':3}


def train(model, device, max_len, num_epochs, batch_size, graph):
	optimizer = torch.optim.Adam([
		dict(params=model.GNN.gconv1.parameters(), weight_decay=5e-4),
		dict(params=model.GNN.gconv2.parameters(), weight_decay=0)
	], lr=0.001)

	criterion = torch.nn.CrossEntropyLoss()
	epochs_stop = 5
	min_loss = np.Inf
	epoch_min_loss = np.Inf
	no_improve = 0
	loss_list = []
	acc_list = []
	start_epoch=1
	checkpoint_path = 'gnn.pt'

	#Check if a checkpoint exists to continue training
	if os.path.isfile('gnn.pt'):
		checkpoint = torch.load('gnn.pt')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		loss = checkpoint['loss']

	model.train()

	for epoch in range(start_epoch, num_epochs):
		for i in range(1, batch_size + 1):
			optimizer.zero_grad()
			data, inds = get_subgraph(graph, batch_size=batch_size)
			adj = data[0]
			print(adj.max())
			print(adj.shape)
			audio =[torchaudio.load(graph.node_id[i])[0] for i in inds ]
			audio = padding_tensor(audio, max_len)
			labels = [emo[graph.y[i]] for i in inds]
			labels=torch.as_tensor(labels)
			audio.to(device)
			labels.to(device)
			x_embedding = model.acoustic_model(audio)
			print(x_embedding.shape)
			#assert data.edge_index.max() < data.num_nodes
			out = model.GNN(x_embedding, adj)
			loss = criterion(out, labels)

			# Track the accuracy
			total = labels.size(0)
			_, predicted = torch.max(out.data, 1)
			correct = (predicted == labels).sum().item()
			print(labels)
			print(predicted)
			print(correct / total)
			acc_list.append(correct / total)

			# Backprop and perform Adam optimization
			loss.backward()
			optimizer.step()

			loss_list.append(loss.item())
			if loss < min_loss:
				min_loss = loss

			if (i + 1) % 2:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
				      .format(epoch , num_epochs, i + 1, total_step, loss.item(),
				              (correct / total) * 100))

			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss,
			}, checkpoint_path)

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
			write_file('gnn_accuracy.txt', acc_list)
			write_file('gnn_loss.txt', loss_list)


##Test Model###
	model.eval()
	correct = 0
	total = 0
	y = []
	y_predicted = []
	with torch.no_grad():
		for data in testloader:
			outputs = model(data)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			y_predicted.append(outputs.numpy())
			y.append(labels.numpy())

	print('Accuracy: %d %%' % (100 * correct / total))
	write_file('gnn_accuracy_test.txt', [(100 * correct / total)])
	model.save('gnn_model.pt')