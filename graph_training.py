import torch
import wandb


def train_graphs(model, train_loader, num_epochs):
	#wandb.init(project="gnn-emotion", entity="arronte")
	lr = 0.001

	wandb.config = {
		"learning_rate": lr,
		"epochs": num_epochs
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss()
	#criterion = F.nll_loss()

	epochs_stop = 3
	min_loss = None
	no_improve = 0
	acc_list = []
	epoch_min_loss = None
	start_epoch=1

	model.train()

	for epoch in range(start_epoch, num_epochs):
		epoch_loss=[]
		print(len(train_loader))
		for graph in train_loader:
			edge_weight = graph.edge_norm * graph.edge_weight
			optimizer.zero_grad()
			labels = graph.y
			out = model(graph.x, graph.edge_index, edge_weight)
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
			epoch_loss.append(loss)

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
		#wandb.log({"loss": loss})
		# wandb.watch(model)


def test_graphs(model, test_loader):
	labels=[]
	predictions=[]
	for graph in test_loader:
		labels.append(graph.y)
		out = model(graph.x, graph.edge_index)
		_, predicted = torch.max(out.data, 1)
		predictions.append(int(predicted))
	total = len(labels)
	correct = (torch.as_tensor(predictions) == torch.as_tensor(labels)).sum().item()
	print((correct / total)*100)