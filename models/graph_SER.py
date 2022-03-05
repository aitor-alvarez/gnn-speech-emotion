from torch import nn
from torch import functional as F

#Combining acoustic representation learning with GNN.

class Prosodic_Graph(nn.Module):
	def __init__(self, acoustic_model, graph_model):
		super().__init__()
		self.acoustic_model = acoustic_model
		self.GNN = graph_model

	#Input of the architecture is a dataset of speech files and a graph data object.
	def forward(self, audio_embeddings, adj):
		x = self.acoustic_model(audio_embeddings)
		x = self.GNN(x, adj)
		return x



