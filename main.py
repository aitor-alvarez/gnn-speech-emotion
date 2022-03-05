import argparse
from train import train
import torch
from models.graph_SER import Prosodic_Graph
import torchaudio
from dataset.dataloader import get_subgraph
from models.gnn import GCNN
from models.speech_representations import ResidualBLSTM, Resblock



def exec(graph_path, batch_size=64, num_epochs=40):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Prosodic_Graph(ResidualBLSTM(Resblock, [2]), GCNN())
	model.to(device)
	graph = torch.load(graph_path)
	max_len = max([torchaudio.load(g)[0].shape[1] for g in graph.node_id ])
	train(model, device, max_len, num_epochs, batch_size)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default = 'patterns/',
                        help='Data path to graph file.')

	parser.add_argument('-b', '--batch_size', type=int, default= 32,
	                    help='Batch size')

	parser.add_argument('-e', '--num_epochs', type=int, default=40,
	                    help='Number of epochs')

	args = parser.parse_args()

	exec(args.data_path, args.batch_size, args.num_epochs)