# preprocess.py
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from nltk.tokenize import word_tokenize

def extract(data):
  """
  function to extract relevant features from the reviews stored in dicts

  data : list(dicts)
  """
  N = len(data)
  text, sentiment = [None]*N, [None]*N
  for i in range(N):
    review = defaultdict(str, data[i])
    text[i] = f"{review['summary']} {review['reviewText']}"
    sentiment[i] = review['sentiment']

  return text, sentiment

def tokenise(X):
  tokenised = []
  for document in tqdm(X):
    tokenised.append(word_tokenize(document))
  return tokenised

def get_token_encoding(X_train):
  word2idx = {"<PAD>": 0}
  idx2word = {0: "<PAD>"}
  idx = 1
  for document in tqdm(X_train):
    for token in document:
      if token not in word2idx:
        word2idx[token] = idx
        word2idx[idx] = token
        idx += 1
  return word2idx, idx2word

def token_encode(X, word2idx):
  res = []
  for document in X:
    doc = []
    for token in document:
      if token not in word2idx:
        doc.append(0)
      else:
        doc.append(word2idx[token])
    res.append(doc)
  return res

def get_label_encoding(y_train):
  uniqs = np.unique(y_train)
  label2idx = {uniqs[i]: i for i in range(len(uniqs))}
  idx2label = {i: uniqs[i] for i in range(len(uniqs))}

  return label2idx, idx2label

def label_encode(y, label2idx):
  return [label2idx[label] for label in y]

def one_hot_encode(X, word2idx):
  N = len(X)
  V = len(word2idx)
  res = np.zeros((N, V))
  for i in tqdm(range(N)):
    document = X[i]
    for j in range(len(document)):
      idx = document[j]
      res[i][idx] = 1
  return res

def preprocess(data):
  text, sentiment = extract(data)
  text = tokenise(text)

  return text, sentiment
