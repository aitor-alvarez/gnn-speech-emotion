import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

class EmotionGraphData(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
		super().__init__(root, transform, pre_transform, pre_filter)

