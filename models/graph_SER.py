from torch import nn
from torch import functional as F

#Combining acoustic representation learning with GNN.

class Prosodic_Graph(nn.Module):
	def __init__(self, graph_model):
		super().__init__()
		self.GNN = graph_model

	#Input of the architecture is a dataset of speech files and a graph data object.
	def forward(self, audio_embeddings, adj):
		x = self.GNN(audio_embeddings, adj)
		return x



