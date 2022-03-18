# baseline.py
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from scripts.baseline import MajorityClass, LogisticRegression
from scripts.feature_ext import get_label_encoding, label_encode
import datetime
import numpy as np

# -------------------------- Global settings

METRICS = [('accuracy', accuracy_score), ('f1', f1_score)]
NOFSPLITS = 5

# -------------------------- Main functions
def evaluate_baseline(data, training=True):
  
  """Evaluate selected baseline models.

  Based on the provided data, train and then evaluate
  corresponding baseline models. // Serves as a wrapper function.

  Parameters
  ----------
  data : dict
    Contains all the needed data to (train) and evaluate given models.
  
  training : bool
    Indicates whether this is a training evaluation or not.
  Returns
  -------
  res : nd array or dict
    If training, then dict is returned with best model, else
    it includes the prediction of the best model. 
  """
  
  if training:

    models = []
    models.append(eval_majority_class(data))

# -------------------------- Utility functions

def select_best_model():
  """
  To be implemented.
  """
  pass
 
def eval_majority_class(data):

  """Evaluates Majority class model.

  The evaluation for this function was implemented
  separately as GridSearch was not used as part
  of the training.

  """
  
  print("> Majority class classifier (cross validated scores)")
  res = {"model_name": "MajorityClass"}
  start = datetime.datetime.now()  
  skf = StratifiedKFold(n_splits=NOFSPLITS)
  X = np.array(data['X_train'])
  y = np.array(data['y_train'])

  for metric, mf in METRICS:
    tmp = []
    for train_index, test_index in skf.split(X, y): 
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      label2idx, _ = get_label_encoding(y_train)
      y_train = label_encode(y_train, label2idx)
      y_test = label_encode(y_test, label2idx)

      model = MajorityClass()
      model.fit(X_train, y_train)
      yhat = model.predict(X_test)
      tmp.append(mf(y_test, yhat))

    s = np.array(tmp).mean()
    print(f">> {metric}: {s}")
    res[metric] = s

  end = datetime.datetime.now()

  diff = end - start
  res["training_time"] = diff
  print(f">> Running time: {diff}\n")

  return res

