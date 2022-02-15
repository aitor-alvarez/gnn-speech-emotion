from torch import nn

#Combining acoustic representation learning with GNN.

class Prosodic_Graph(nn.Module):
	def __init__(self, acoustic_model, graph_model):
		super().__init__()
		self.acoustic_model = acoustic_model
		self.GNN = graph_model

	#Input of the architecture is a dataset of speech files and a network adjacency list.
	def forward(self, audio_file, adj):
		x = self.acoustic_model(audio_file)
		x = self.GNN(x, adj)
		return x
