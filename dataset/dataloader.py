import torch
import os

#Load graphs and audio samples
def train_data_loader(data_path):
		subs = os.listdir(data_path)
		graphs=[]
		for sub in subs:
			if os.path.isdir(data_path + sub+'/'):
				subdir = os.listdir(data_path + sub+'/')
				for sd in subdir:
					if os.path.isdir(data_path + sub+'/'+sd+'/'):
						files = os.listdir(data_path + sub+'/'+sd+'/')
						for f in files:
							if f.endswith('.pt'):
								graphs.append(torch.load(data_path + sub + '/' + sd + '/' + f))
		return graphs



def padding_tensor(sequences):
    """
    input=list of tensors
    """
    num = len(sequences)
    max_len = max([s.size(1) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(1)
        out_tensor[i, :length] = tensor
    return out_tensor