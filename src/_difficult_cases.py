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
from sklearn.metrics import accuracy_score

def main():
  # loading data
  total = timer()
  start = working_on('Loading Train and Dev Data')
  X_train, y_train, X_dev, y_dev = get_data(stage='one_hot_encoded', split=['train', 'dev'])
  word2idx, idx2label = get_encodings(['word2idx', 'idx2label'])

  finished('Loading Train and Dev Data', timer() - start)

  # scale data
  start = working_on('Scaling Data')
  X_train, X_dev = X_train.toarray(), X_dev.toarray()
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_dev  scaler.transform(X_dev)
  finished('Scaling Data', timer()-start)

  # logistic regression baseline
  start = working_on('Predicting on Validation Split')
  clf = LogisticRegression() # default params
  clf.fit(X_train, y_train)

  preds = clf.predict(X_dev)
  print(f">> Validation Accuracy: {accuracy_score(y_dev, preds)}")
  finished('Predicting on Validation Split', timer()-start)

  # saving difficult cases
  start = working_on('Finding Difficult Cases')
  path = 'data/difficult_cases'
  os.makedirs(path) if not os.path.exists(path) else None
  with open(f"{path}/difficult_cases.txt") as outfile:
    for i in range(len(y_dev)):
      if preds[i] != y_dev[i]:
        outfile.write('yay\n')
  finished('Finding Difficult Cases', timer()-start)

  finished('Entire Difficult Cases Pipeline', timer()-total)

if __name__ == "__main__":
  main()
