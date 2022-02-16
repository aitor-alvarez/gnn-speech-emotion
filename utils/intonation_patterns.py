import os
import numpy as np
import parselmouth
from pydub import  AudioSegment
from utils.gapbide import Gapbide

#Functions to extract speech utterances and intonation contours

def generate_dataset(input_dir, output_dir):
	patterns, files = get_patterns(input_dir)



#Process audio files and return intervallic contours.
def get_patterns(audio_dir):
	fqs, files = get_f0_praat(audio_dir)
	contours, inds = get_interval_contour(fqs)
	pattern_length = 5
	patterns= Gapbide(contours, 6, 0, 0, pattern_length).run()
	return patterns, files


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
					continue
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