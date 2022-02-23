import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset.dataloader import audio_batch


def train(model, device, optimizer, trainloader, batch_size, num_epochs):
	model.train()
	train_loss = 0

	for batch in range(1, num_epochs+1):
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
			optimizer.step()


def test(model, device):
	model.eval()