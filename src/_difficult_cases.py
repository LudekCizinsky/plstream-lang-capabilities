# _difficult_cases.py
# script to generate, save and evaluate difficult cases for
# out baseline classifier
from scripts.utils import output, working_on, finished
output('Loading Modules')

from scripts.utils import load_model, get_data, get_encodings

import os
import json
import spacy
import checklist
import argparse
import numpy as np
import warnings
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.expect import Expect
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re

warnings.filterwarnings("ignore")

def mft1(n):
  editor = Editor()
  reviews = editor.template(
  'This {musicterm} {mask} is really not {pos_adj}',
  musicterm=['album', 'artist', 'producer', 'song', 'tune', 'work'],
  pos_adj=['good', 'nice', 'enjoyable', 'likeable', 'great'],
  remove_duplicates=True,
  nsamples=n).data

  return reviews, ['negative']*n

def mft2(n):
  editor = Editor()
  reviews = editor.template(
  'I had high expectations for this {musicterm}. But, it ended up being {mask}.',
  musicterm=['album', 'artist', 'producer', 'song', 'tune', 'work'],
  remove_duplicates=True,
  nsamples=n).data

  return reviews, ['negative']*n

def typo_test(n):
  reviews, sentiments = get_data(stage='extracted', split='train')
  
  i = 0
  typos = []
  sentiment = []
  while i<n:
    idx = np.random.randint(len(reviews))
    review = reviews[idx]

    typo = Perturb.perturb([review], Perturb.add_typos)['data'][0][1]
    if typo != review:
      typos.append(typo)
      sentiment.append(sentiments[idx])
      i += 1

  return typos, sentiment

def remove_stars(n):
  removed_stars = []
  removed_sentiment = []
  reviews, sentiments = get_data(stage='extracted', split='train')
  pattern = '[a-zA-Z]+ [Ss]tars{0,1} ' 
  num = 0
  for review, sentiment in zip(reviews,sentiments):
    if re.match(pattern, review):
      replaced = re.sub(pattern, '', review)
      removed_stars.append(replaced)
      removed_sentiment.append(sentiment)
      num += 1
      
    if num > n-1:
      break

  return removed_stars,removed_sentiment
      
def prank_test(n):

  reviews, sentiments = get_data(stage='extracted', split='train')

  pos_prank = ' Did you really believe I do not like it? No way! It was a masterpiece!'
  neg_prank = ' Now for real, it is actually terrible, and I could not dislike it more.'
  
  i = 0
  pranks = []
  sentiment = []
  while i<n:
    idx = np.random.randint(len(reviews))
    review = reviews[idx]
    sent = sentiments[idx]
    
    if sent == "negative":
      pranks.append(review + pos_prank)
      sentiment.append("positive")
    elif sent == "positive":
      pranks.append(review + neg_prank)
      sentiment.append("negative")
    else:
      raise ValueError("Incorrect sentiment.")
      
    i += 1

  return pranks, sentiment

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-L','--load', action='store_true', help='load difficult cases otherwise, generate them')
  args = parser.parse_args()  
  LOAD = args.load

  total = timer()
  if not LOAD:
    start = working_on('Generating Difficult Cases')

    # generate difficult cases using checklist library
    X_mft1, y_mft1 = mft1(25)
    X_mft2, y_mft2 = mft2(25)
    X_typo, y_typo = typo_test(25)
    X_stars, y_stars = remove_stars(25)
    X_prank, y_prank =  prank_test(25)

    finished('Generating Difficult Cases', timer() - start)

    TESTS = {
      'mft1': (X_mft1, y_mft1),
      'mft2': (X_mft2, y_mft2),
      'typos': (X_typo, y_typo),
       'truth_at_the_end': (X_prank, y_prank),
      }
    
    difficult_cases = []
    for test_type, data in TESTS.items():
      for review, sentiment in zip(data[0], data[1]):
        case = {}
        case['reviewText'] = review
        case['sentiment'] = sentiment
        case['category'] = test_type
        difficult_cases.append(case)


    start = working_on('Save Difficult Cases')
    path = 'data/difficult_cases'
    os.makedirs(path) if not os.path.exists(path) else None
    with open(f"{path}/difficult_cases.json", "w") as f:
      tmp = [json.dumps(case) for case in difficult_cases]
      res = "\n".join(tmp) 
      f.write(res)

    # store them in dict save as json object into
    # data/difficult_cases
    finished('Save Difficult Cases', timer() - start)

  else:
    difficult_cases = []
    with open('data/difficult_cases/difficult_cases.json', 'r') as f:
      for line in f:
        difficult_cases.append(json.loads(line))
  
  # load the model and the difficult cases, predict them and
  # report on relevant statistics (should be as low as
  # possible) in the different categories
  start = working_on('Evaluating Baseline Performance')
  N = len(difficult_cases)
  X = [None]*N
  y = [None]*N
  categories = [None]*N
  for i in range(N):
    X[i] = difficult_cases[i]['reviewText']
    y[i] = difficult_cases[i]['sentiment']
    categories[i] = difficult_cases[i]['category']

  model = load_model('data/models/logistic_regression.pkl')
  label2idx = get_encodings(['label2idx'])

  y = [label2idx[label] for label in y]

  preds = model.predict(X)
  print(f"Accuracy Score: {accuracy_score(y, preds)}")
  print(classification_report(y, preds))
  print(f"F1 Score: {f1_score(y, preds)}")

  finished('Evaluating Baseline Performance', timer() - start)
  
  # finished entire pipeline
  finished('Entire Difficult Cases Pipeline', timer() - total)

if __name__ == "__main__":
  main()
