import argparse
from train import train, test
import torch
from models.graph_SER import Prosodic_Graph
from dataset.dataloader import train_data_loader, audio_batch
from torch.functional import F



def exec(data_path, batch_size, num_epochs):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Prosodic_Graph().to(device)
	optimizer = torch.optim.Adam(lr=0.0001)
	graphs, audio = train_data_loader(data_path).to(device)
	train(model, device, optimizer, graphs, audio, batch_size, num_epochs)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default = 'patterns/',
                        help='Data path directory.')

	parser.add_argument('-b', '--batch_size', type=int, default= 32,
	                    help='Batch size')

	parser.add_argument('-e', '--num_epochs', type=int, default=None,
	                    help='Number of epochs')

	exec(parser.data_path, parser.batch_size, parser.num_epochs)