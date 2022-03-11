import torch
from dataset.dataloader import get_subgraph, write_file
import os
from torch.optim.lr_scheduler import ExponentialLR


def train(model, graph, num_epochs):
	optimizer = torch.optim.Adam([
		dict(params=model.gconv1.parameters(), weight_decay=5e-4),
		dict(params=model.gconv2.parameters(), weight_decay=0)
	], lr=0.01)
	scheduler = ExponentialLR(optimizer, gamma=0.9)
	criterion = torch.nn.CrossEntropyLoss()
	epochs_stop = 10
	min_loss = None
	no_improve = 0
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
		print(epoch)
		optimizer.zero_grad()
		labels = graph.y
		weights = graph.edge_weight
		out = model(graph.x, graph.edge_index, weights)
		loss = criterion(out, labels)

		# Track the accuracy
		total = labels.size(0)
		_, predicted = torch.max(out.data, 1)
		correct = (predicted == labels).sum().item()
		print(correct / total)
		acc_list.append(correct / total)

		# Backprop and perform Adam optimization
		loss.backward()
		optimizer.step()
		print(loss)

		# Scheduler decay
		scheduler.step()

		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
		}, checkpoint_path)

		if min_loss == None:
			min_loss = loss
		elif loss < min_loss:
			min_loss = loss
			no_improve = 0
		else:
			no_improve += 1
		if no_improve == epochs_stop:
			break


def test(model_path, model, graph):
	labels = graph.y
	weights = graph.edge_weight
	total = labels.size(0)
	checkpoint = torch.load(model_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	out = model(graph.x, graph.edge_index, weights)
	_, predicted = torch.max(out.data, 1)
	correct = (predicted == labels).sum().item()
	print(correct / total)