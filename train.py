import torch
import torchaudio
from dataset.dataloader import padding_tensor


emo ={'ang':0, 'hap':1, 'neu':2, 'sad':3}


def train(model, device, optimizer, trainloader, num_epochs):
	model.to(device)
	model.train()
	train_loss = 0
	criterion = torch.nn.CrossEntropyLoss()

	for e in range(1, num_epochs+1):
		for i, data in enumerate(trainloader, 0):
			optimizer.zero_grad()
			graph = data
			audio = [torchaudio.load(file)[0] for a in data.y for file in a]
			audio = padding_tensor(audio)
			labels = [emo[d[d.find('/')+1:d.find('/')+4]] for a in data.y for d in a]
			labels=torch.as_tensor(labels)
			audio.to(device)
			labels.to(device)
			x_embedding = model.acoustic_model(audio)
			output = model.GNN(x_embedding, graph)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()

	#model.eval()





