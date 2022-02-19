# Prosody-Aware Graph Neural Networks for Robust Speech Emotion Recognition

### Create prosodic patterns trianing set

Execute the following command:
```
python preprocess.py -a '~/IEMOCAP/train/'

```

The IEMOCAP /train directory should follow this structure:

 - ang
 - sad
 - hap
 - neu
  
Where each emotion label is a subdirectory of the train/ directory.

