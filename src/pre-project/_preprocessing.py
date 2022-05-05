# _preprocessing.py
# script that contains entire preprocessing pipeline

import numpy as np
from scipy.sparse import save_npz
import os
import json
from tqdm import tqdm
from timeit import default_timer as timer

from scripts.utils import get_data
from scripts.utils import output, working_on, finished
from scripts.feature_ext import (
    _extract, 
    _tokenise,
    _get_token_encoding, 
    _get_label_encoding, 
    _token_encode, 
    _label_encode, 
    _one_hot_encode
    )

def main():
  total = timer() # global timer

  # stage: raw
  raw_train = get_data(stage='raw', split='train')
  raw_dev = get_data(stage='raw', split='dev')
  raw_test = get_data(stage='raw', split='test')
  finished('Loading Raw Data')

  # stage: extracted 
  start = working_on('Extracting Relevant Features and Labels')

  reviews_train, sentiment_train = _extract(raw_train)
  reviews_dev, sentiment_dev = _extract(raw_dev)
  reviews_test, sentiment_test = _extract(raw_test) # sentiment masked

  extracted = {
      'train': (reviews_train, sentiment_train),
      'dev': (reviews_dev, sentiment_dev),
      'test': (reviews_test, sentiment_test)
      }

  for split, data in tqdm(extracted.items()):
    path = f'data/processed/extracted/'
    os.makedirs(path) if not os.path.exists(path) else None
    for i in range(len(data)):
      if not i: name = f'X_{split}.txt'
      else: name = f'y_{split}.txt'

      with open(path+name, 'w') as outfile:
        N = len(data[i])
        for j in range(N):
          outfile.write(f"{data[i][j].strip()}\n")

  print('Saved tokenised files to data/processed/extracted')
  finished('Extracting Relevant Features and Labels', time=timer()-start)

  # stage: tokenised
  start = working_on('Tokenising Reviews')

  tokenised_train = _tokenise(reviews_train)
  tokenised_dev = _tokenise(reviews_dev)
  tokenised_test = _tokenise(reviews_test)

  tokenised = {
      'train': (tokenised_train, sentiment_train),
      'dev': (tokenised_dev, sentiment_dev),
      'test': (tokenised_test, sentiment_test)
      }

  for split, data in tqdm(tokenised.items()):
    path = f'data/processed/tokenised/'
    os.makedirs(path) if not os.path.exists(path) else None
    for i in range(len(data)):
      if not i: name = f'X_{split}.tsv'
      else: name = f'y_{split}.tsv'

      with open(path+name, 'w') as outfile:
        N = len(data[i])
        for j in range(N):
          if i == 0:
            out = '\t'.join(data[i][j])
          else: 
            out = data[i][j]
          outfile.write(f"{out}\n")
  print('Saved Extracted Files to data/processed/tokenised')
  finished('Tokenising Reviews', timer()-start)

  # stage: int-encoded
  start = working_on('Integer Encoding Reviews and Labels')
  word2idx, idx2word = _get_token_encoding(tokenised_train)
  label2idx, idx2label = _get_label_encoding(sentiment_train)

  encodings = {
      'word2idx': word2idx,
      'idx2word': idx2word,
      'label2idx': label2idx,
      'idx2label': idx2label
      }

  for name, mapping in encodings.items():
    path = 'data/encodings/'
    name = f'{name}.json'
    os.makedirs(path) if not os.path.exists(path) else None
    with open(path + name, 'w') as outfile:
      json.dump(mapping, outfile)
  print('Saved mappings to data/encodings')

  # integer encode
  X_train  = _token_encode(tokenised_train, word2idx)
  y_train = _label_encode(sentiment_train, label2idx)

  X_dev= _token_encode(tokenised_dev, word2idx)
  y_dev = _label_encode(sentiment_dev, label2idx)

  X_test = _token_encode(tokenised_test, word2idx)
  y_test = [-1 for _ in range(len(X_test))]

  int_encoded = {
      'train': (X_train, y_train),
      'dev': (X_dev, y_dev),
      'test': (X_test, y_test)
      }

  for split, data in tqdm(int_encoded.items()):
    path = f'data/processed/int_encoded/'
    os.makedirs(path) if not os.path.exists(path) else None
    for i in range(len(data)):
      if not i: name = f'X_{split}.csv'
      else: name = f'y_{split}.csv'

      with open(path+name, 'w') as outfile:
        N = len(data[i])
        for j in range(N):
          if i == 0:
            out = ','.join(map(str, data[i][j]))
          else: 
            out = str(data[i][j])
          outfile.write(f"{out}\n")
  print('saved integer encoded reviews and labels to data/processed/int_encoded')
  finished('Integer Encoding Reviews and Labels', time=timer()-start)


  working_on('One-Hot Encoding Reviews')
  start = timer()

  X_train = _one_hot_encode(X_train, word2idx=word2idx)
  X_dev = _one_hot_encode(X_dev, word2idx=word2idx)
  X_test = _one_hot_encode(X_test, word2idx=word2idx)

  one_hot = {
      'train': (X_train, y_train),
      'dev': (X_dev, y_dev),
      'test': (X_test, y_test)
      }

  # save one hot encoded reviews
  path = "data/processed/one_hot_encoded"
  os.makedirs(path) if not os.path.exists(path) else None
  for split, data in tqdm(one_hot.items()):
    X, y = data
    save_npz(f"{path}/X_{split}.npz", X)
    with open(f"{path}/y_{split}.csv", 'w') as outfile:
      N = len(y)
      for i in range(N):
        outfile.write(f"{y[i]}\n")
  print('Saved One-Hot Encoded files to data/processed/one_hot_encoded')
  finished('One-Hot Encoding Reviews', timer()-start)

  finished('Entire Preprocessing Pipeline', timer()-total)

if __name__ == "__main__":
  main()
