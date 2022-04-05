# _senti_word_net.py
# script to play around with SentiWordNet 3.0

import os
import csv
import numpy as np
from tqdm import tqdm
from nltk.corpus import sentiwordnet as swn
from sklearn.metrics import accuracy_score

# custom scripts
from scripts.utils import get_data

def main():
  X_train, y_train = get_data(stage='tokenised', split='dev')

  preds = []
  for review in tqdm(X_train):
    pred = 0
    for token in review:
      synsets = list(swn.senti_synsets(token))
      avg_pos_score = np.mean([ss.pos_score() for ss in synsets]) if synsets else 0
      avg_neg_score = np.mean([ss.neg_score() for ss in synsets]) if synsets else 0
      pred += avg_pos_score
      pred -= avg_neg_score
    if pred > 0:
      preds.append('positive')
    elif pred < 0:
      preds.append('negative')
    else: 
      preds.append('neutral')

  print(accuracy_score(y_train, preds))

if __name__ == "__main__":
  main()
