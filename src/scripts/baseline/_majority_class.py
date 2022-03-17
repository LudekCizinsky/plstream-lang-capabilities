# _majority_class.py
import numpy as np
from tensorboard import summary

class MajorityClass():
  def __init__(self):
    """the majority class classifier"""
    self._majority = None
    self._probs = None

  def fit(self, X, y):
    identifier, counts = np.unique(np.array(y), return_counts=True) # Counting the occurances of 0 and 1
    self._probs = counts/len(y)
    self._majority = identifier[np.argmax(counts)]

  def predict(self, X):
    return np.empty(len(X)).fill(self._majority)

  def predict_probability(self, X):
    return np.array([self._probs for _ in range(len(X))])


