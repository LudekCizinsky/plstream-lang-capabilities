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
import numpy as np
import warnings
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.expect import Expect
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

def language_test(n):
  editor = Editor(language='german')
  reviews = editor.template(
    'This {musicterm} {mask} is really not {pos_adj}',
    musicterm=['album', 'artist', 'producer', 'song', 'tune', 'work'],
    pos_adj=['good', 'nice', 'enjoyable', 'likeable', 'great'],
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

def negation_test(n):
  reviews, sentiments = get_data(stage='extracted', split='train')

  # load spacy
  nlp = spacy.load('en_core_web_sm')
  
  i = 0
  negated = []
  sentiment = []
  while i<n:
    idx = np.random.randint(len(reviews))
    review = reviews[idx]

    negation = Perturb.remove_negation(nlp(review)) 
    print(negation)

    if negation != review:
      negated.append(negation)
      sentiment.append(sentiments[idx])
      i += 1

  return negated, sentiment

def main():
  total = timer()
  if False:
    start = working_on('Generating Difficult Cases')

    # generate difficult cases using checklist library
    X_mft1, y_mft1 = mft1(200)
    X_mft2, y_mft2 = mft2(200)
    X_typo, y_typo = typo_test(200)
    # X_neg, y_neg = negation_test(200)
    X_lang, y_lang = language_test(200)

    finished('Generating Difficult Cases', timer() - start)

    TESTS = {
        'mft1': (X_mft1, y_mft1),
        'mft2': (X_mft2, y_mft2),
        'typos': (X_typo, y_typo),
        # 'negation': (X_neg, y_neg),
        'language': (X_lang, y_lang),
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
