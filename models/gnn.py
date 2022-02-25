from torch import nn
from torch_geometric.nn import GCNConv, GINConv
from torch.functional import F


#GCNN baseline
class GCNN(nn.Module):

	def __init__(self, in_features, num_classes=4):
		super().__init__()
		self.gconv1 = GCNConv(in_features, 512)
		self.gconv2 = GCNConv(512, num_classes)
		self.relu = nn.LeakyReLU()


	def forward(self, in_feat, graph):
		x, edge_index, = in_feat, graph.edge_index
		x = self.conv1(x, edge_index)
		x = self.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		x = F.log_softmax(x, dim=1)
		return x









