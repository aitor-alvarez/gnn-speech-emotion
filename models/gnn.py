import torch
from torch import nn
from torch.functional import F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATv2Conv, DeepGCNLayer, GENConv
from torch.nn import LayerNorm, Linear, ReLU



#GCNN
class GCNN(nn.Module):

	def __init__(self):
		super().__init__()
		self.gconv1 = GCNConv(128, 64)
		self.gconv2 = GCNConv(64, 32)
		self.gconv3 = GCNConv(32, 4)
		self.relu = nn.LeakyReLU()


	def forward(self, x_embeddings, edge_index, weights):
		x = self.gconv1(x_embeddings, edge_index, weights)
		x = self.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.gconv2(x, edge_index, weights)
		x = self.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.gconv3(x, edge_index, weights)
		out = F.softmax(x, dim=1)
		return out


#GCNN with attention
class AttGCNN(nn.Module):

	def __init__(self):
		super().__init__()
		self.gconv1 = GATv2Conv(128, 64, heads=4, dropout=0.6)
		self.gconv2 = GATv2Conv(64*4, 4, heads=1, concat=False, dropout=0.6)
		self.relu = nn.LeakyReLU()


	def forward(self, x, edge_index, weights):
		x = F.dropout(x, p=0.6, training=self.training)
		x = self.relu(self.gconv1(x, edge_index))
		x = F.dropout(x, p=0.6, training=self.training)
		x = self.gconv2(x, edge_index)
		return F.softmax(x, dim=1)



class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, graph):
        super().__init__()

        self.node_encoder = Linear(graph.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(graph.edge_weight.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, graph.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr.shape
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)