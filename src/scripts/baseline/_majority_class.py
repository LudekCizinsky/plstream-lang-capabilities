# _majority_class.py
import numpy as np
from tensorboard import summary

class MajorityClass():
  def __init__(self, matrix):
    """the majority class classifier"""
    self.matrix = matrix
    return

  def majority(self, matrix):
    sentiment = matrix[-1]                                        # Acess the column with the sentiment 
    identifier, counts = np.unique(sentiment, return_counts=True) # Counting the occurances of 0 and 1
    count_sent = dict(zip(identifier, counts))                    # Zip the counts into a dictionary
    find_majority = max(count_sent, key=count_sent.get)           # Find the majority count from the dict
    print(find_majority)
    return find_majority