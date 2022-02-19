import argparse
from utils.intonation_patterns import generate_dataset, build_corpus
import os

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-b', '--build_corpus', type=str, default=None,
	                    help='Initial arrangement of the dataset by emotion. Provide the root path for the IEMOCAP dataset.')

	parser.add_argument('-a', '--generate_dataset', type=str, default = None,
	                        help='The audio directory where the training set by emotion is located.')

	parser.add_argument('-l', '--label', type=str, default = None,
	                        help='Dataset label')

	args = parser.parse_args()

	if args.build_corpus:
		build_corpus(args.build_corpus)
	if args.generate_dataset:
		subs = os.listdir(args.generate_dataset)
		for s in subs:
			if '.DS_Store' not in s:
				print(s)
				generate_dataset(args.generate_dataset, s)
	else:
		print("Please provide the arguments needed. Check the help -h to see the arguments available.")



if __name__ == '__main__':
    main()

