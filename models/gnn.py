import torch
from torch import nn
from torch.functional import F
from torch_geometric.nn import GCNConv, GATv2Conv, DeepGCNLayer, GENConv, GraphConv


#GCNN
class GCNN(nn.Module):

	def __init__(self):
		super().__init__()
		self.gconv1 = GCNConv(128, 128)
		self.gconv2 = GCNConv(128, 4)
		self.relu = nn.LeakyReLU()


	def forward(self, x_embeddings, edge_index, weights):
		x = self.gconv1(x_embeddings, edge_index, weights)
		x = self.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.gconv2(x, edge_index, weights)
		out = F.softmax(x, dim=1)
		return out


#GCNN with attention
class AttGCNN(nn.Module):

	def __init__(self):
		super().__init__()
		self.gconv1 = GATv2Conv(128, 128, heads=4, dropout=0.4)
		self.gconv2 = GATv2Conv(128*4, 4, heads=1, concat=False, dropout=0.4)
		self.relu = nn.LeakyReLU()


	def forward(self, x, edge_index, weights):
		x = self.relu(self.gconv1(x, edge_index))
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.gconv2(x, edge_index)
		return F.softmax(x, dim=1)