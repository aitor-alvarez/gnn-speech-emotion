import torch
from torch.functional import F
from dataset.dataloader import get_subgraph, write_file
import os
from torch.optim.lr_scheduler import ExponentialLR
import wandb


def train(model, train_loader, graph, num_epochs):
	wandb.init(project="gnn-emotion", entity="arronte")
	lr = 0.005

	wandb.config = {
		"learning_rate": lr,
		"epochs": num_epochs
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
	#scheduler = ExponentialLR(optimizer, gamma=0.9)
	#criterion = torch.nn.CrossEntropyLoss()
	#criterion = F.nll_loss()

	epochs_stop = 5
	min_loss = None
	no_improve = 0
	acc_list = []
	epoch_min_loss = None
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
		epoch_loss=[]
		for graph in train_loader:
			optimizer.zero_grad()
			labels = graph.y
			weights = graph.edge_weight
			out = model(graph.x, graph.edge_index, weights)
			loss = F.nll_loss(out, labels)

			# Track the accuracy
			total = labels.size(0)
			_, predicted = torch.max(out.data, 1)
			correct = (predicted == labels).sum().item()
			acc_list.append(correct / total)

			# Backprop and perform Adam optimization
			loss.backward()
			optimizer.step()
			print(loss)
			epoch_loss.append(loss)

			# Scheduler decay
			#scheduler.step()

			#torch.save({
			#	'epoch': epoch,
			#	'model_state_dict': model.state_dict(),
			#	'optimizer_state_dict': optimizer.state_dict(),
			#	'loss': loss,
			#}, checkpoint_path)

		### Epoch check ###
		e_loss = sum(epoch_loss) / len(epoch_loss)
		if epoch_min_loss == None:
			epoch_min_loss = e_loss
		elif e_loss < epoch_min_loss:
			epoch_min_loss = e_loss
			no_improve = 0
		else:
			no_improve += 1
		if no_improve == epochs_stop:
			break

		# Visualization
		wandb.log({"loss": loss})
		# wandb.watch(model)




def test(model, graph):
	labels = graph.y
	weights = graph.edge_weight
	total = labels.size(0)
	out = model(graph.x, graph.edge_index, weights)
	_, predicted = torch.max(out.data, 1)
	correct = (predicted == labels).sum().item()
	print((correct / total)*100)
