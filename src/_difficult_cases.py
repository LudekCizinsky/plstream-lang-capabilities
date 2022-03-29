# _difficult_cases.py
# script to generate, save and evaluate difficult cases for
# out baseline classifier
from scripts.utils import output, working_on, finished
output('Loading Modules')

from scripts.utils import load_model, get_data, get_encodings

import os
import json
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
  total = timer()
  start = working_on('Generating Difficult Cases')
  # generate difficult cases using checklist library
  difficult_cases = []

  finished('Generating Difficult Cases', timer() - start)

  start = working_on('Save Difficult Cases')
  # store them in dict save as json object into
  # data/difficult_cases
  finished('Save Difficult Cases', timer() - start)
  
  # load the model and the difficult cases, predict them and
  # report on relevant statistics (should be as low as
  # possible) in the different categories
  start = working_on('Evaluating Baseline Performance')
  N = len(difficult_cases)
  X_train = [None]*N
  y_train = [None]*N
  categories = [None]*N
  for i in range(N):
    X_train[i] = difficult_cases[i].reviewText
    y_train[i] = difficult_cases[i].sentiment
    categories[i] = difficult_cases[i].category

  model = load_model('data/models/logistic_regression.pkl')
  label2idx = get_encodings(['label2idx'])

  preds = model.predict(X_train)
  print(f"Accuracy Score: {accuracy_score(y_true, preds)}")
  print(f"F1 Score: {f1_score(y_true, preds)}")

  finished('Evaluating Baseline Performance', timer() - start)
  
  # finished entire pipeline
  finished('Entire Difficult Cases Pipeline', timer() - total)

if __name__ == "__main__":
  main()
