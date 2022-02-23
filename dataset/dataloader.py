import torch
import os


def train_data_loader(data_path):
		subs = os.listdir(data_path)
		graphs=[]
		audio=[]
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
								audio_files.append((data_path + sub+'/'+sd+'/'+f, sub))
						audio.append(audio_files)
		return graphs, audio


def audio_batch(loader_data, audio):
	audio_out=[]
	for l in loader_data:
		for a in audio:
			if l in a[0]:
				audio_out.append(a)
	return audio_out
