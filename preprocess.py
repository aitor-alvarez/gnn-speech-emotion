import argparse
from utils.intonation_patterns import generate_graph , build_corpus, create_contours
import os
from itertools import chain

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-b', '--build_corpus', type=str, default=None,
	                    help='Initial arrangement of the dataset by emotion. Provide the root path for the IEMOCAP dataset.')

	parser.add_argument('-a', '--generate_dataset', type=str, default = None,
	                        help='The audio directory where the training set by emotion is located.')

	parser.add_argument('-g', '--generate_graph', type=str, default = None,
	                        help='IEMOCAP dataset directory.')


	args = parser.parse_args()

	if args.build_corpus:
		build_corpus(args.build_corpus)
	if args.generate_graph:
		subs = os.listdir(args.generate_graph)
		print("Generating graph...")
		contours_list=[]
		files_list=[]
		for s in subs:
			if os.path.isdir(args.generate_graph+s):
				contours, files, pitches, inds = create_contours(args.generate_graph, s)
				contours_list.append(contours)
				files_list.append(files)
		generate_graph(list(chain.from_iterable(contours_list)), list(chain.from_iterable(files_list)))
	else:
		print("Please provide the arguments needed. Check the help command -h to see the arguments available.")




if __name__ == '__main__':
    main()

