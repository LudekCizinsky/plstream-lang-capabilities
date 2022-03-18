import numpy as np


class MajorityClass:

  def __init__(self): 
    self._majority = None
    self._probs = None

  def fit(self, X, y):
    identifier, counts = np.unique(np.array(y), return_counts=True)
    self._probs = counts/len(y)
    self._majority = identifier[np.argmax(counts)]

  def predict(self, X):
    yhat = np.empty(len(X))
    yhat.fill(self._majority)
    return yhat

  def predict_probability(self, X):
    return np.array([self._probs for _ in range(len(X))])

