#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scripts.utils import output, working_on, finished
output('Loading Modules')

from scripts.utils import get_data, get_encodings
from scripts.feature_ext import get_training_data, get_test_data
from scripts.evaluation import evaluate_baseline
from timeit import default_timer as timer
import datetime
import os
import json

def main():
  total = timer() # global timer
  start = working_on('Loading Data')

  # get extracted data (extracted and tokenised)
  X_train, y_train, X_dev, y_dev = get_data(stage='extracted', split=['train', 'dev']) 

  # encode labels to integer format
  word2idx, label2idx = get_encodings(which=['word2idx', 'label2idx'])
  y_train = [label2idx[label] for label in y_train]
  y_dev = [label2idx[label] for label in y_dev]

  # combine training and development set
  X_train = X_train + X_dev
  y_train = y_train + y_dev
  finished('Loading Data', timer() - start)

  # model training/ evaluation
  start = working_on('Training and Evaluating Baseline Models')
  models = evaluate_baseline({"X_train": X_train, "y_train": y_train})
  finished('Training and Evaluating Baseline Models', timer() - start)

  # testing models
  start = working_on('Evaluating on Test Data')
  raw = get_data(stage='raw', split='test')
  X_test, _  = get_data(stage='extracted', split='test')

  # untokenise test
  X_test = [' '.join(review) for review in X_test]

  preds = evaluate_baseline(
      {"X_test": X_test, 'models': models, 'label2idx': label2idx},
      training=False
  )
  finished('Evaluating on Test Data', timer() - start)
  
  # save test predictions
  start = working_on('Saving Test Predictions in Original Format')
  for r, y in zip(raw, preds):
    r["sentiment"] = y    
  file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
  path = f"data/predictions/{file_name}.yhat"
  with open(path, "w") as f:
    tmp = [json.dumps(r) for r in raw]
    res = "\n".join(tmp) 
    f.write(res)
  print(f"Predictions saved to: {path}")
  finished('Saving Test Predictions in Original Format', timer() - start)

  finished('Entire Pipeline', timer() - total)

if __name__ == "__main__":
  main()
