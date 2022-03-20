import os
import torch
import numpy as np
import parselmouth
from pydub import AudioSegment
from utils.gapbide import Gapbide
import pandas as pd
from utils.process_file import create_dictionary, create_nodes_dictionary
import uuid
from utils.MaximaPatterns import MaximalPatterns
import networkx as nx
from torch_geometric.utils import from_networkx


#Functions to extract speech utterances and intonation contours

#IEMOCAP labels used
emotions=['ang', 'hap', 'neu', 'sad', 'exc']

#We build the corpus by creating directories by emotion to be used by a dataloader
def build_corpus(iemocap_dir):
	subpath='/Train/'
	csv_files = [csv for csv in os.listdir(iemocap_dir+subpath) if csv.endswith('.csv')]
	print("Creating the corpus...")
	for c in csv_files:
		df= pd.read_csv(iemocap_dir+subpath + '/' +c)
		for row in df.itertuples():
			if row[4] in emotions:
				if os.path.isdir(iemocap_dir+subpath +row[4]):
					os.rename(iemocap_dir+subpath+subpath +row[3]+'.wav', iemocap_dir+subpath+ '/' +row[4]+'/'+row[3]+'.wav' )
				else:
					os.mkdir(iemocap_dir + subpath + row[4])
					os.rename(iemocap_dir + subpath + subpath + row[3]+'.wav',
					          iemocap_dir + subpath  + row[4] + '/' + row[3]+'.wav')
	print("corpus completed")


#Create the dataset of patterns, extracting the audio and graphs for each emotional utterance
def generate_dataset(audio_dir, emo, train=False):
	if train == True: sub='train/'
	if train == False: sub='test/'
	contours, files, pitches, inds= create_contours(audio_dir, emo)
	pattern_length = 8
	#Gapbide(contours, 12, 0, 0, pattern_length, filename).run()
	#MaximalPatterns(filename+'_intervals.txt', filename + '_maximal.txt').execute()
	dictionary = create_dictionary('patterns/'+sub+emo+'_maximal.txt')
	path_out_audio='patterns/'+sub+emo+'/'
	create_audio_samples(dictionary, contours, files, pitches, inds, audio_dir+emo+'/', path_out_audio)


def create_contours(audio_dir, emo):
	fqs, files, pitches = get_f0_praat(audio_dir + emo + '/')
	contours, inds = get_interval_contour(fqs)
	return contours, files, pitches, inds


###Creates a graph based on the prosodic similarity of the speech utterances.
def generate_graph(contours, files):
	dictionary = create_nodes_dictionary('patterns/train/')
	G = nx.Graph()
	node_list=[]
	for d in dictionary:
		nodes = []
		for i, c in enumerate(contours):
			nodename = files[i]
			if len(d) > len(c):
				continue
			else:
				sub = find_sublist(d, c)
			if sub:
				nodes.append(nodename)
				G.add_node(nodename, node_id=nodename, y=nodename[nodename.rfind('/')-3:nodename.rfind('/')])
		g = nx.Graph()
		g.add_nodes_from(nodes)
		sg= create_graph(g)
		G.add_edges_from(sg.edges, weight=1.00)
		node_list.append(nodes)
	graph = add_edge_attributes(G, node_list)
	gp= from_networkx(graph)
	torch.save(gp, 'patterns/graph.pt')


def add_edge_attributes(G, nodes):
	for e in G.edges:
		for n in nodes:
			if e[0] and e[1] in n:
				if 'weight' in G[e[0]][e[1]]:
					G[e[0]][e[1]]['weight'] +=1
	return G


#Takes as the input a dictionary of (intonation) patterns and contours and slices audio files based on the patterns
# contained in the dictionary. At the same time it saves the co-occurences of patterns in an adjacency list to
#create a graph.

def create_audio_samples(dictionary, contours, files, pitches, inds, path, path_out_audio):
	for i, c in enumerate(contours):
		G = nx.Graph()
		path2 = path+files[i]
		for d in dictionary:
			if len(d) > len(c):
				continue
			else:
				sub = find_sublist(d, c)
			if sub:
				for s in sub:
					name = files[i].replace('.wav', '_')+str(uuid.uuid4())+'.wav'
					ini = inds[i][s[0]][0]+1
					end = inds[i][s[1]][0]+1
					slice_audio(pitches[i].get_time_from_frame_number(ini), pitches[i].get_time_from_frame_number(end), path2, name, path_out_audio)
					G.add_node(name, y=path_out_audio+name)
		if G.number_of_nodes()>0:
			graph = create_graph(G)
			graph = from_networkx(graph)
			torch.save(graph, path_out_audio +files[i].replace('.wav', '')+ '.pt')


def slice_audio(slice_from, slice_to, path, audio_file, path_out):
	audio = AudioSegment.from_wav(path)
	try:
		seg = audio[slice_from * 1000:slice_to * 1100]
		seg.set_channels(2)
		seg.export(path_out+audio_file, format="wav", bitrate="192k")
	except:
		print(f"ERROR PROCESSING AUDIO FILE: {path}")


#extract f0 from Parselmouth Praat function
def get_f0_praat(audio_dir):
	files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
	pitches = [parselmouth.Sound(audio_dir+f).to_pitch(pitch_floor=75.0, pitch_ceiling=650.0) for f in files]
	fqs = [pitch.kill_octave_jumps().selected_array['frequency'] for pitch in pitches]
	return fqs, files, pitches


#return a list of intervallic distances between F0 points expressed in cents
def get_interval_contour(fqs):
	contours = []
	inds= []
	for f in fqs:
		contour = []
		ind = []
		for i in range(len(f)-1):
			if i < len(f):
				if f[i] == 0 or f[i+1] == 0:
					continue
				else:
					dist = 1200 * np.log2(f[i+1]/f[i])
					dist = get_interval(dist)
					contour.append(dist)
					ind.append((i, i+1))
		contours.append(contour)
		inds.append(ind)
	return contours, inds


def find_sublist(s,l):
    result=[]
    sll=len(s)
    for ind in (i for i,e in enumerate(l) if e==s[0]):
        if l[ind:ind+sll]==s:
            result.append((ind,ind+sll-1))
    return result


def get_interval(dist):
	i = abs(dist)
	if i < 50:
		if dist < 0:
			return '-1'
		elif dist == 0:
			return '0'
		else:
			return '1'
	elif i >= 50 and i < 100:
		if dist < 0:
			return '-2'
		else:
			return '2'
	elif i >= 100 and i < 150:
		if dist < 0:
			return '-3'
		else:
			return '3'
	elif i >= 150 and i < 200:
		if dist < 0:
			return '-4'
		else:
			return '4'
	elif i >= 200 and i < 250:
		if dist < 0:
			return '-5'
		else:
			return '5'
	elif i >= 250 and i < 300:
		if dist < 0:
			return '-6'
		else:
			return '6'
	elif i >= 300 and i < 350:
		if dist < 0:
			return '-7'
		else:
			return '7'
	elif i >= 350 and i < 400:
		if dist < 0:
			return '-8'
		else:
			return '8'
	elif i >= 400 and i < 450:
		if dist < 0:
			return '-9'
		else:
			return '9'
	elif i >= 450 and i < 500:
		if dist < 0:
			return '-10'
		else:
			return '10'
	elif i >= 500 and i < 550:
		if dist < 0:
			return '-11'
		else:
			return '11'
	else:
		if dist < 0:
			return '-12'
		else:
			return '12'


def create_graph(G, type='cycle'):
	if type == 'cycle':
		e = nx.cycle_graph(G)
	G.add_edges_from(e.edges)
	return G