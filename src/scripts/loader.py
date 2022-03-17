# loader.py
import os
import sys
import gzip
import json

def load_data(split='dev'):
  """
  Loading specified raw data split as a list of dictionaries.
  Each dictionary represents a single review.
  """
  if split not in ['train', 'dev', 'test']:
    raise ValueError("split not defined. choose from ['train', 'dev', 'test']")
  if split == 'test':
    split = 'test_masked'

  data = []
  for line in gzip.open(f'data/classification/music_reviews_{split}.json.gz'):
    review_data = json.loads(line)
    data.append(review_data)

  return data
