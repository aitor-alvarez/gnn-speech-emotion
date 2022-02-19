import networkx as nx
import os

#Create a graph based on the prosodic patterns extracted, in this case patterns/
#It returns a networkx graph object
def create_graph(patterns_dir):
	G=nx.Graph()
	for d in os.listdir(patterns_dir):
		if os.path.isdir(patterns_dir+d+'/'):
			files = os.listdir(patterns_dir+d+'/')
			for f in files:
				files.remove(f)
				adj = []
				g = nx.Graph()
				adj.append(f)
				for k in files:
					if f.rfind('_') == k.rfind('_'):
						print('same')
						adj.append(k)
						files.remove(k)
				g.add_nodes_from(adj)
				g = nx.complete_graph(g)
				G.update(g)
	return G




