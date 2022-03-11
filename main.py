import torch
import argparse
from node_train import train, test
import torchaudio
from models.gnn import GCNN
from models.speech_representations import ResidualBLSTM, Resblock
from pretrain import pretrain
from dataset.dataloader import padding_tensor
import os
from torch.utils.data import DataLoader


def exec(graph_path='patterns/train/graph_weights.pt', speech_model_path='pretrained/speech_representation.pt', num_epochs=200, complete=False):
	if complete == True:
		graph = torch.load(graph_path)
		graph = complete_graph_with_speech_features(speech_model_path, graph)
		torch.save(graph, 'patterns/graph.pt')
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		graph = torch.load(graph_path)
		model = GCNN()
		model.to(device)
		train(model, graph, num_epochs)
		test('gnn.pt', model, torch.load('patterns/test/graph.pt'))


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
	elif args.data_path:
		exec(args.data_path, args.speech_model, args.batch_size, args.num_epochs)
	else:
		print("please provide valid arguments. See -h for help.")