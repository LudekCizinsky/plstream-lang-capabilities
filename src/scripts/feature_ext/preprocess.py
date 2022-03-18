# preprocess.py
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from nltk.tokenize import word_tokenize
from .loader import load_data
import nltk
nltk.download('punkt')

# -------------------------- Main functions

def get_training_data():
  """Load in memory training data.
  
  For efficiency, we load training data
  separately from test data to not waste RAM.

  Returns
  -------
  X_train : list
    2D list where each item represents tokenized review, each token is a string.

  y_train : list
    1D list where each item indicates sentiment label which is a string.
  """
  
  raw = [
      ('train', load_data(split='train')),
      ('dev', load_data(split='dev'))
  ]

  X, y = list(), list() 
  
  for _, d in raw:
    text, sentiment = extract(d)
    text = tokenise(text)
    X.extend(text)
    y.extend(sentiment)

  return X, y

# -------------------------- Utility functions

def extract(data):
  """Function to extract relevant features from the reviews.

  Parameters
  ----------
  data : list(dictionaries)
    Each dictionary represents review

  Returns
  -------
  text : list
    Each item is a summary with actual review.

  sentiment : list
    List with value being either 'positive' or 'negative'
  """

  N = len(data)
  text, sentiment = [None]*N, [None]*N
  for i in range(N):
    review = defaultdict(str, data[i])
    text[i] = f"{review['summary']} {review['reviewText']}"
    sentiment[i] = review['sentiment']

  return text, sentiment

def tokenise(X):
  """Tokenise reviews using nltk.

  Parameters
  ----------
  X : list
    Each item is a string representing a review.

  Returns
  -------
  tokenised : list
    Nested list where each inner list holds tokens for given review. 

  Notes
  -----
  To see the details of the tokenisation, please visit
  `nltk docs <https://www.nltk.org/api/nltk.tokenize.html>`_
  """

  tokenised = []
  for document in tqdm(X):
    tokenised.append(word_tokenize(document))
  return tokenised

def get_token_encoding(X_train):
  """Map tokens to idx and vice versa.

  Parameters
  ----------
  X_train : list
    Nested list where each inner items represents tokenised review.

  Returns
  -------
  word2idx : dict
    Token to int.

  idx2word : dict
    Int to token.

  Notes
  -----
  It is important to run this function ONLY on training data.
  """

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
  """Map tokens to corresponding id.
  
  Parameters
  ----------
  X : list
    Nested list where each inner items represents tokenised review.
  word2idx: dict
    Map according which the encoding of tokens should be done.

  Returns
  -------
  res : list
    Nested list where each inner item represents tokenized review,
    however instead of strings, we have ids.

  Notes
  -----
  If it encounters not known token, it assigns it with an id = 0. 
  """

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
  """Map string labels to ids.
  
  Parameters
  ----------
  y_train : list
    List with strings representing sentiment of the review.

  Returns
  -------
  label2idx : dict
    Mapping of a label to id.

  idx2label : dict
    Mapping of an id to label.
  """
  uniqs = np.unique(y_train)
  label2idx = {uniqs[i]: i for i in range(len(uniqs))}
  idx2label = {i: uniqs[i] for i in range(len(uniqs))}

  return label2idx, idx2label

def label_encode(y, label2idx):
  """Map labels to ids.
  
  y : list
    List with strings representing sentiment of the review.

  idx2label : dict
    Mapping of an id to label.

  Returns
  -------
  res : list
    List with ids corresponding to labels.
  """
  res = [label2idx[label] for label in y]
  return res

def count_vectorizer(X, word2idx, binary=True):
  """Create a bag of words vector.

  Parameters
  ----------
  X : list
    Nested list where each items is a tokenised review and each
    token is represent by an id.

  word2idx : dict
    Token to int.

  binary : bool
    If true, then j-position of the i-th vector
    indicates whether the given token is present
    or not. Otherwise, counts are provided of the
    given token.

  Returns
  -------
  res : N-dim array
    N x |V| array where N is number of reviews and |V|
    is size of the vocabulary.
  """

  N = len(X)
  V = len(word2idx)
  res = np.zeros((N, V))
  for i in tqdm(range(N)):
    tokens_idx, counts = np.unique(X[i], returns_count=True)
    for j, token_id in enumerate(tokens): 
      if binary:
        res[i][token_id] = 1
      else:
        res[i][token_id] = counts[j]
  return res

