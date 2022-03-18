# loader.py
import os
import sys
import gzip
import json
import math

def load_data(split='dev', subsample=math.inf):
  """
  Loading specified raw data split as a list of dictionaries.
  Each dictionary represents a single review.
  """
  if split not in ['train', 'dev', 'test']:
    raise ValueError("split not defined. choose from ['train', 'dev', 'test']")
  if split == 'test':
    split = 'test_masked'

  data = []
  for i, line in enumerate(gzip.open(f'data/classification/music_reviews_{split}.json.gz')):
    review_data = json.loads(line)
    data.append(review_data)
    
    if i >= subsample:
      break

  return data
