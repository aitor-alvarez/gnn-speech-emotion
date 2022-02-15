import os
import numpy as np
import parselmouth
from pydub import  AudioSegment

#Functions to extract speech utterances and intonation contours and to plot them with their acoustic features

#Segment speech utterances based on
def extract_speech_utterances(dir_path, slice_path):
	files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
	pitches = [parselmouth.Sound(dir_path+f).to_pitch() for f in files ]
	fqs = [pitch.selected_array['frequency'] for pitch in pitches]
	k = 0
	for fq in fqs:
		nonzero = fq.nonzero()[0]
		diff = np.diff(nonzero)
		skip_inds = np.where(diff>1)[0]
		newInd = nonzero[0]
		filename = files[k].replace('.wav', '')
		for s in skip_inds:
			try:
				dist  = pitches[k].get_time_from_frame_number(nonzero[s+1])- pitches[k].get_time_from_frame_number(nonzero[s])
				if dist >= 0.25:
					slice_audio(pitches[k].get_time_from_frame_number(newInd), pitches[k].get_time_from_frame_number(nonzero[s]), slice_path, filename+'_'+str(s)+'.flac', dir_path+filename+'.flac')
					newInd = nonzero[s+1]
			except:
				print("error")
		dist = pitches[k].get_time_from_frame_number(len(pitches[k])) - pitches[k].get_time_from_frame_number(nonzero[-1])
		if dist >= 0.25 and dist<0.5:
			slice_audio(pitches[k].get_time_from_frame_number(nonzero[-1]), pitches[k].get_time_from_frame_number(len(pitches[k])), slice_path, filename+'_'+str(s+1)+'.flac', dir_path+filename+'.flac')
		k +=1
	print("segmentation completed")


def slice_audio(slice_from, slice_to, path, name, audio_file):
	audio = AudioSegment.from_wav(audio_file)
	try:
		seg = audio[slice_from * 1000:slice_to * 1100]
		seg.set_channels(2)
		seg.export(path+name, format="wav", bitrate="192k")
	except:
		print("NO")


#extract f0 from Parselmouth Praat function
def get_f0_praat(audio_dir):
	files = [f for f in os.listdir(audio_dir) if f.endswith('.flac')]
	pitches = [parselmouth.Sound(audio_dir + f).to_pitch(pitch_floor=75.0, pitch_ceiling=650.0) for f in files]
	fqs = [pitch.kill_octave_jumps().selected_array['frequency'] for pitch in pitches]
	return fqs, files


#return a list of intervallic distances between F0 points expressed in cents
def get_interval_contour(fqs):
	contours = []
	inds=[]
	for f in fqs:
		contour = []
		ind = []
		for i in range(len(f)-1):
			if i < len(f):
				if f[i] == 0 or f[i+1] == 0:
					pass
				else:
					dist = 1200 * np.log2(f[i+1]/f[i])
					dist = get_interval(dist)
					contour.append(dist)
					ind.append((i, i+1))
		contours.append(contour)
		inds.append(ind)
	return contours, inds


def get_interval(dist):
	i = abs(dist)
	if i < 50:
		return '0'
	elif i >= 50 and i < 150:
		if dist < 0:
			return '-1'
		else:
			return '1'
	elif i >= 150 and i < 250:
		if dist < 0:
			return '-2'
		else:
			return '2'
	elif i >= 150 and i < 250:
		if dist < 0:
			return '-2'
		else:
			return '2'
	elif i >= 250 and i < 350:
		if dist < 0:
			return '-3'
		else:
			return '3'
	elif i >= 350 and i < 450:
		if dist < 0:
			return '-4'
		else:
			return '4'
	elif i >= 450 and i < 550:
		if dist < 0:
			return '-5'
		else:
			return '5'
	elif i >= 550 and i < 650:
		if dist < 0:
			return '-6'
		else:
			return '6'
	else:
		if dist < 0:
			return '-7'
		else:
			return '7'