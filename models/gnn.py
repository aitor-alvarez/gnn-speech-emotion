import torch
from torch import nn
from torch.functional import F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool


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



#Graph classification with avg. pooling
class GNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,
	             normalize=False, lin=True):
		super(GNN, self).__init__()

		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()

		self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
		self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

		self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
		self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

		self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
		self.bns.append(torch.nn.BatchNorm1d(out_channels))

	def forward(self, x, adj):

		for step in range(len(self.convs)):
			x = self.bns[step](F.relu(self.convs[step](x, adj)))

		return x


class DiffPool(torch.nn.Module):
	def __init__(self):
		super(DiffPool, self).__init__()

		self.gnn1_pool = GNN(128, 128, 128)
		self.gnn1_embed = GNN(128, 128, 128)

		self.gnn2_pool = GNN(128, 128, 128)
		self.gnn2_embed = GNN(128, 128, 128, lin=False)

		self.lin1 = torch.nn.Linear(128, 64)
		self.lin2 = torch.nn.Linear(64, 4)


	def forward(self, data):
		x = data.x
		adj = data.edge_index
		batch = data.batch
		x = self.gnn1_pool(x, adj)
		x = self.gnn1_embed(x, adj)
		x = global_mean_pool(x, batch)
		x = self.lin1(x)
		x = self.lin2(x)
		return F.log_softmax(x, dim=-1)


class GraphCNN(nn.Module):

	def __init__(self):
		super().__init__()
		self.gconv1 = GCNConv(128, 128)
		self.gconv2 = GCNConv(128, 4)
		self.gconv3 = GCNConv(128, 4)
		self.relu = nn.LeakyReLU()


	def forward(self, data):
		x = data.x
		adj = data.edge_index
		batch = data.batch
		x = self.gconv1(x, adj)
		x = self.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.gconv2(x, adj)
		x = global_mean_pool(x, batch)
		out = F.softmax(x, dim=1)
		return out