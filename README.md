# Prosody-Aware Subgraph Embeddings for Robust Speech Emotion Recognition

### Create subgraphs based on the prosodic content of the utterances

Execute the following commands to generate the prosodic patterns for train and test sets:
```
python preprocess.py -a '~/IEMOCAP/train/'

python preprocess.py -a '~/IEMOCAP/test/'

```
Generate graph with the following command:

```
python preprocess.py -g '~/IEMOCAP/train/'

```

The IEMOCAP /train directory should follow this structure:

 - ang
 - sad
 - hap
 - neu
  
Where each emotion label is a subdirectory of the train/ directory. The ouput results in a dataset of audio segments corresponding with each of the maximal patterns. A graph per utterance is also created. Audio of patterns and graphs will be stored in the directory patterns/ that should exist in the root directory.

### Pretrain the speech model

Use the following command where -b and -e parameters are bact size and epochs respectively. 

```
python main.py -ptrain '~/IEMOCAP/train/' -ptest '~/IEMOCAP/test/' -b 128 -e 100

```
