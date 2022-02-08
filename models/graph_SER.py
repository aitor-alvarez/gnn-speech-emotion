from .speech_representations import ResidualBLSTM
from .gnn import GCNN
from torch import nn

#Combining acoustic representation learning with graph rep. learning.

class Prosodic_Graph(nn.Module):
	def __init__(self, acoustic_model, graph_model):
		super().__init__()
		self.acoustic_model = acoustic_model
		self.GNN = graph_model

	def forward(self, audio_spec):
		x = self.acoustic_model(audio_spec)
		x = self.GNN(x)

		return x
