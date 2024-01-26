import torch
import wandb


def train_graphs(model, train_loader, num_epochs):
	lr = 0.005

	wandb.config = {
		"learning_rate": lr,
		"epochs": num_epochs
	}

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
	criterion = torch.nn.CrossEntropyLoss()
	#criterion = F.nll_loss()

	epochs_stop = 5
	min_loss = None
	no_improve = 0
	acc_list = []
	epoch_min_loss = None
	start_epoch=1

	model.train()

	for epoch in range(start_epoch, num_epochs):
		epoch_loss=[]
		for graph in train_loader:
			optimizer.zero_grad()
			labels = graph.label
			out = model(graph)
			loss = criterion(out, labels)

			# Track the accuracy
			total = labels.size(0)

			# Backprop and perform Adam optimization
			loss.backward()
			optimizer.step()
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
		print(e_loss)
		print(no_improve)
		# Visualization
		#wandb.log({"loss": loss})
		# wandb.watch(model)


def test_graphs(model, test_loader):
	model.eval()
	torch.no_grad()
	labels=[]
	predictions=[]
	with torch.no_grad():
		for graph in test_loader:
			out = model(graph)
			_, predicted = torch.max(out.data, 1)
			labels.append(graph.label)
			predictions.append(int(predicted))
		total = len(labels)
		correct = (torch.as_tensor(predictions) == torch.as_tensor(labels)).sum().item()
		print((correct / total)*100)
