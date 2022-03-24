# _difficult_cases.py
# script to extract the analyse the wrong predictions of the
# baseline model
from scripts.utils import output, working_on, finished
output('Loading Modules')

from scripts.utils import get_data, get_encodings
from scripts.baseline import LogisticRegression

import os
import numpy as np
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
  # loading data
  total = timer()
  start = working_on('Loading Train and Dev Data')

  # loading textual dev reviews
  dev_reviews, _ = get_data(stage='extracted', split='dev')
  dev_reviews = [' '.join(review) for review in dev_reviews]

  # loading training and dev one hot encoded
  X_train, y_train, X_dev, y_dev = get_data(stage='one_hot_encoded', split=['train', 'dev'])
  word2idx, idx2label = get_encodings(['word2idx', 'idx2label'])

  finished('Loading Train and Dev Data', timer() - start)

  # scale data
  start = working_on('Scaling Data')
  scaler = StandardScaler(with_mean=False)
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_dev = scaler.transform(X_dev)
  finished('Scaling Data', timer()-start)

  # logistic regression baseline
  start = working_on('Predicting on Validation Split')
  clf = LogisticRegression(C=.05, max_iter=1000) # default params
  clf.fit(X_train, y_train)

  preds = clf.predict(X_dev)
  print(f">> Validation Accuracy: {accuracy_score(y_dev, preds)}")
  print(f">> False Positives :\n{confusion_matrix(y_dev, preds)[0, 1]}") 
  print(f">> False Negatives :\n{confusion_matrix(y_dev, preds)[1, 0]}") 
  finished('Predicting on Validation Split', timer()-start)
  return

  # saving difficult cases
  start = working_on('Finding Difficult Cases')
  path = 'data/difficult_cases'
  os.makedirs(path) if not os.path.exists(path) else None
  with open(f"{path}/difficult_cases.txt", "w") as outfile:
    for i in range(len(y_dev)):
      if preds[i] != y_dev[i]:
        review = dev_reviews[i]
        pred = 'positive' if int(preds[i]) else 'negative'
        true = 'positive' if int(y_dev[i]) else 'negative'
        outfile.write(f"{review} - {pred} ({true})\n")
  finished('Finding Difficult Cases', timer()-start)

  finished('Entire Difficult Cases Pipeline', timer()-total)

if __name__ == "__main__":
  main()
