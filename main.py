import argparse
from train import train
import torch
from models.graph_SER import Prosodic_Graph
from dataset.dataloader import train_data_loader
from torch_geometric.loader import DataLoader
from models.gnn import GCNN
from models.speech_representations import ResidualBLSTM, Resblock


def exec(data_path, batch_size=16, num_epochs=40):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Prosodic_Graph(ResidualBLSTM(Resblock, [2]), GCNN)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	graphs = train_data_loader(data_path)
	graph_data = DataLoader(graphs, batch_size=batch_size, shuffle=True)
	train(model, device, optimizer, graph_data, num_epochs)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default = 'patterns/',
                        help='Data path directory.')

	parser.add_argument('-b', '--batch_size', type=int, default= 32,
	                    help='Batch size')

	parser.add_argument('-e', '--num_epochs', type=int, default=None,
	                    help='Number of epochs')

	exec(parser.data_path, parser.batch_size, parser.num_epochs)