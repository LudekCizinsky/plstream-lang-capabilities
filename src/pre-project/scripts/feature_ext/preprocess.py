# preprocess.py
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')

# -------------------------- Public functions
def get_test_data(label2idx):
  """Load in memory testing data.
  
  For efficiency, we load training data
  separately from test data to not waste RAM.
  
  Parameters
  ----------
  label2idx : dict
    Mapping from label to id.

  Returns
  -------
  X : list
    2D list where each item represents tokenized review, each token is a string.

  raw : list
    1D list where each item is a dictionary representing meta info about the
    review instance.
  """
  
  
  raw = load_data(split='test')
  X, _ = _extract(raw) 

  return X, raw


def get_training_data():
  """Load in memory training data.
  
  For efficiency, we load training data
  separately from test data to not waste RAM.

  Returns
  -------
  X : list
    2D list where each item represents tokenized review, each token is a string.

  y : list
    1D list where each item indicates sentiment label which is a string.

  label2idx : dict
    Mapping from label to id.
  """
  
  raw = [
      ('train', load_data(split='train')),
      ('dev', load_data(split='dev'))
  ]

  X, y = list(), list() 
  
  label2idx = None
  for _, d in raw:
    text, sentiment = _extract(d)
    sentiment, label2idx  = label_encode(sentiment, label2idx=label2idx)
    X.extend(text)
    y.extend(sentiment)

  return X, y, label2idx

def label_encode(y, label2idx=None):

  """Encodes given labels.

  Parameters
  ----------
  y : list
    List with strings representing sentiment of the review.

  label2idx : dict
    Mapping of a label to id. If provided returns the same dict.
    Should be provided only for labels from dev or test data.

  Returns
  -------
  y_encoded : list
    List with ids corresponding to labels.
 
  label2idx : dict
    Mapping of a label to id. If provided returns the same dict.
  """

  if label2idx is None:
    label2idx, idx2label = _get_label_encoding(y)

  y_encoded = _label_encode(y, label2idx)

  return y_encoded, label2idx


def count_vectorizer(X, word2idx=None, binary=True):
  """Create a bag of words vector.

  Parameters
  ----------
  X : Iterable
    Nested iterable where each inner item represents tokenised review  and
    tokens are UNencoded, i.e., strings.
    

  word2idx : dict
    Token to int. Should be provided only for dev or test data.

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

  word2idx: dict
    Mapping from word to id.
  """


  if word2idx is None:
    word2idx, idx2word = _get_token_encoding(X)

  X_encoded = _token_encode(X, word2idx)

  bow = _count_vectorizer(X_encoded, word2idx, binary)
  
  return bow, word2idx

# -------------------------- Private functions

def _extract(data):
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
  for i in tqdm(range(N)):
    review = defaultdict(str, data[i])
    text[i] = f"{review['summary']} {review['reviewText']}".replace('\n', '')
    sentiment[i] = review['sentiment']

  return text, sentiment

def _tokenise(data):
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
  N = len(data)
  tokenised = [None]*N
  for i in tqdm(range(N)):
    tokenised[i] = word_tokenize(data[i])
  return tokenised

def _get_token_encoding(X_train):
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

def _token_encode(X, word2idx):
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

def _get_label_encoding(y_train):
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

def _label_encode(y, label2idx):
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

def _one_hot_encode(X, word2idx, binary=True):
  """Create a bag of words vector.

  Parameters
  ----------
  X : Iterable
    Nested iterable where each items is a tokenised review and each
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
  freq = defaultdict(int) 

  for i in tqdm(range(N)):
    for token_id in X[i]: 
      if binary:
        freq[(i, token_id)] = 1
      else:
        freq[(i, token_id)] = 1

  row, col = zip(*freq.keys())
  data = [c for c in freq.values()]

  one_hot = coo_matrix((data, (row, col)), shape=(N, V))

  return one_hot
