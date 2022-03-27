# Prosody-Aware Graph Neural Networks for Speech Emotion Recognition

### Create subgraphs based on the prosodic content of the utterances

Execute the following commands to generate the prosodic patterns for train and test sets:
```
python preprocess.py -a '~/IEMOCAP/train/'

python preprocess.py -a '~/IEMOCAP/test/'

```
Generate graph with the following command:

```
python preprocess.py -g '~/IEMOCAP/train/'
python preprocess.py -g '~/IEMOCAP/test/'

```

The IEMOCAP /train directory should follow this structure:

 - ang
 - sad
 - hap
 - neu
  
Where each emotion label is a subdirectory of the train/ directory. 

### Pretrain the speech model

Use the following command where -b and -e parameters are bact size and epochs respectively. 

```
python main.py -ptrain '~/IEMOCAP/train/' -ptest '~/IEMOCAP/test/' -b 128 -e 100

```
