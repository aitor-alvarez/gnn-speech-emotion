import torch
import argparse
from node_train import train, test
from graph_training import train_graphs, test_graphs
import torchaudio
from models.gnn import GCNN, AttGCNN, DiffPool
from models.speech_representations import ResidualBLSTM, Resblock
from pretrain import pretrain
from dataset.dataloader import padding_tensor, graph_loader
import os
from torch.utils.data import DataLoader
from torch_geometric.loader import GraphSAINTNodeSampler
from torch_geometric.loader import DataLoader as DL


#Training batches of graphs
def graph_training(dir='patterns/', num_epochs=200):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train = dir + 'train/'
	test = dir + 'test/'
	model = DiffPool()
	model.to(device)
	train_dataloader = graph_loader(train)
	train_dataloader = DL(train_dataloader, batch_size=32, shuffle=True)
	test_dataloader = graph_loader(test)
	test_dataloader =  DL(test_dataloader, batch_size=1, shuffle=False)
	train_graphs(model, train_dataloader, num_epochs=num_epochs)
	test_graphs(model, test_dataloader)


def node_training(graph_path='patterns/train/graph_weights.pt', speech_model_path='pretrained/speech_representation.pt', num_epochs=200, complete=False):
	if complete == True:
		graph = torch.load(graph_path)
		graph = complete_graph_with_speech_features(speech_model_path, graph)
		torch.save(graph, 'patterns/test_graph.pt')
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		graph = torch.load(graph_path)
		model = AttGCNN()
		model.to(device)
		train_loader = GraphSAINTNodeSampler (graph, batch_size=925, num_steps=30, sample_coverage=100)
		train(model, train_loader, graph, num_epochs)
		test(model, torch.load('patterns/test/graph.pt'))


def update_graphs_speech_features(speech_model_path='pretrained/speech_representation.pt', train=False):
	emo = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
	if train == True: dir = 'patterns/train/'
	if train == False: dir = 'patterns/test/'
	speech_model = ResidualBLSTM(Resblock, [2])
	if os.path.isfile(speech_model_path):
		checkpoint = torch.load(speech_model_path)
		speech_model.load_state_dict(checkpoint['model_state_dict'])
	# Remove the last classification layer in the model
	speech_model.classify = torch.nn.Identity()
	speech_model.eval()

	subs = os.listdir(dir)
	for s in subs:
		if os.path.isdir(dir + s + '/'):
			files = os.listdir(dir + s + '/')
			for f in files:
				if f.endswith('.pt'):
					graph = torch.load(dir + s + '/'+f)
					graph.x = get_speech_representations(speech_model, graph.y)
					graph.label = emo[graph.y[0][graph.y[0].rfind('/')-3:graph.y[0].rfind('/')]]
					torch.save(graph, dir + s + '/'+f)


#add speech features to x in graph object
def complete_graph_with_speech_features(speech_model_path, graph):
	emo = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
	speech_model = ResidualBLSTM(Resblock, [2])
	if os.path.isfile(speech_model_path):
		checkpoint = torch.load(speech_model_path)
		speech_model.load_state_dict(checkpoint['model_state_dict'])
	# Remove the last classification layer in the model
	speech_model.classify = torch.nn.Identity()
	speech_model.eval()
	graph.x = get_speech_representations(speech_model, graph.node_id)
	graph.y = torch.as_tensor([emo[y] for y in graph.y])
	graph.edge_weight = graph.weight
	return graph


def get_speech_representations(speech_model, data, max_len=510560):
	embeddings=[]
	data= [torchaudio.load(d)[0] for d in data]
	data = padding_tensor(data, max_len)
	data = DataLoader(data, batch_size=1, shuffle=False)
	with torch.no_grad():
		for audio in data:
			outputs = speech_model(audio)
			embeddings.append(outputs)
	out = torch.cat(embeddings)
	return out


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-ptrain', '--audio_path1', type=str, default='iemocap/train/',
	                    help='Pretrain the model. Provide IEMOCAP training data path.')

	parser.add_argument('-ptest', '--audio_path2', type=str, default='iemocap/test/',
	                    help='Pretrain the model. Provide IEMOCAP test data path.')

	parser.add_argument('-d', '--data_path', type=str, default = 'patterns/',
                        help='Data path to graph file.')

	parser.add_argument('-m', '--speech_model', type=str, default='pretrained/',
	                    help='path to the directory where the pretrained acoustic model is located.')

	parser.add_argument('-b', '--batch_size', type=int, default= 32,
	                    help='Batch size')

	parser.add_argument('-e', '--num_epochs', type=int, default=40,
	                    help='Number of epochs')

	args = parser.parse_args()

	if args.audio_path1 and args.audio_path2:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = ResidualBLSTM(Resblock, [2])
		pretrain(model, device, args.num_epochs, args.batch_size, args.audio_path1, args.audio_path2)
	elif args.data_path and args.num_epochs:
		node_training(args.data_path)
		if args.num_epochs:
			node_training(args.data_path, num_epochs=args.num_epochs)
	else:
		print("please provide valid arguments. See -h for help.")