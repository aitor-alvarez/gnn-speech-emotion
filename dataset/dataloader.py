import torch
import os
import torchaudio


def train_data_loader(data_path):
		subs = os.listdir(data_path)
		graphs=[]
		audio=[]
		labels=[]
		for sub in subs:
			if os.path.isdir(data_path + sub+'/'):
				subdir = os.listdir(data_path + sub+'/')
				for sd in subdir:
					if os.path.isdir(data_path + sub+'/'+sd+'/'):
						files = [f for f in data_path + sub+'/'+sd+'/']
						graph_files = [torch.load(data_path + sub+'/'+sd+'/'+g) for g in files if g.endswith('.pt')]
						audio_files = [torchaudio.load(data_path + sub+'/'+sd+'/'+d) for d in files if g.endswith('.wav')]
						label = [sub]*len(audio_files)
						graphs.append(graph_files)
						audio.append(audio_files)
						labels.append(label)
		return graphs, audio, labels