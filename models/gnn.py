from torch import nn
from torch.functional import F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GINConv


#GCNN baseline
class GCNN(nn.Module):

	def __init__(self):
		super().__init__()
		self.gconv1 = GCNConv(128, 64)
		self.gconv2 = GCNConv(64, 4)
		self.relu = nn.LeakyReLU()
		self.fc = Linear(64, 4)


	def forward(self, x_embeddings, edge_index):
		print(x_embeddings.shape)
		x = self.gconv1(x_embeddings, edge_index)
		x = self.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.gconv2(x, edge_index)
		out = F.softmax(x, dim=1)
		return out