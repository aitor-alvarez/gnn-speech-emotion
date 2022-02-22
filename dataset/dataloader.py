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
						files = os.listdir(data_path + sub+'/'+sd+'/')
						audio_files=[]
						for f in files:
							if f.endswith('.pt'):
								graphs.append(torch.load(data_path + sub + '/' + sd + '/' + f))
							elif f.endswith('.wav'):
								audio_files.append(torchaudio.load(data_path + sub+'/'+sd+'/'+f))
						audio.append(audio_files)
						labels.append([sub]*len(audio_files))
		return graphs, audio, labels