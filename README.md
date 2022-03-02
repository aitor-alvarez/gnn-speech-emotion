# Prosody-Aware Subgraph Embeddings for Robust Speech Emotion Recognition

### Create prosodic patterns training set and subgraph for each utterance

Execute the following command:
```
python preprocess.py -a '~/IEMOCAP/train/'

```

The IEMOCAP /train directory should follow this structure:

 - ang
 - sad
 - hap
 - neu
  
Where each emotion label is a subdirectory of the train/ directory. The ouput results in a dataset of audio segments corresponding with each of the maximal patterns. A graph per utterance is also created. Audio of patterns and graphs will be stored in the directory patterns/ that should exist in the root directory.

