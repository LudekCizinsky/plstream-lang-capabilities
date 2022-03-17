# _majority_class.py
import numpy as np
from tensorboard import summary

class MajorityClass():
  def __init__(self, Y):
    """the majority class classifier"""
    self.Y = Y
    return

  def majority(self, Y):
    #Y = matrix[-1]                                       # Use this if Y is in a matrix 
    identifier, counts = np.unique(np.array(Y), return_counts=True) # Counting the occurances of 0 and 1
    count_sent = dict(zip(identifier, counts))            # Zip the counts into a dictionary
    find_majority = max(count_sent, key=count_sent.get)   # Find the majority count from the dict
    print(find_majority)
    return 