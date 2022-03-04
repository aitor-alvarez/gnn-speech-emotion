import argparse
from train import train
import torch
from models.graph_SER import Prosodic_Graph
import torchaudio
from torch_geometric.loader import DataLoader
from models.gnn import GCNN
from models.speech_representations import ResidualBLSTM, Resblock


def exec(graph_path, batch_size=32, num_epochs=40):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Prosodic_Graph(ResidualBLSTM(Resblock, [2]), GCNN())
	model.to(device)
	graphs = torch.load(graph_path)
	max_len = max([torchaudio.load(g.node_id)[0].shape[1] for g in graphs ])
	trainloader= DataLoader(graphs, batch_size=batch_size, shuffle=True)
	train(model, device, trainloader, max_len, num_epochs)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default = 'patterns/',
                        help='Data path directory.')

	parser.add_argument('-b', '--batch_size', type=int, default= 32,
	                    help='Batch size')

	parser.add_argument('-e', '--num_epochs', type=int, default=40,
	                    help='Number of epochs')

	args = parser.parse_args()

	exec(args.data_path, args.batch_size, args.num_epochs)