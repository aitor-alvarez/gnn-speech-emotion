import torch
import os
import torchaudio

#Load graphs and audio samples
def data_loader(data_path):
		subs = os.listdir(data_path)
		graphs=[]
		max_len=[]
		for sub in subs:
			if os.path.isdir(data_path + sub+'/'):
				subdir = os.listdir(data_path + sub+'/')
				for sd in subdir:
					if os.path.isdir(data_path + sub+'/'+sd+'/'):
						files = os.listdir(data_path + sub+'/'+sd+'/')
						for f in files:
							if f.endswith('.pt'):
								graphs.append(torch.load(data_path + sub + '/' + sd + '/' + f))
							elif f.endswith('.wav'):
								max_len.append(torchaudio.load(data_path + sub + '/' + sd + '/' + f)[0].shape[1])
		return graphs, max(max_len)



def padding_tensor(sequences, max_len):
    """
    input=list of tensors
    """
    num = len(sequences)
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(1)
        out_tensor[i, :length] = tensor
    return out_tensor