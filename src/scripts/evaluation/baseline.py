# baseline.py
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from scripts.baseline import MajorityClass, LogisticRegression
from scripts.feature_ext import label_encode
from scripts.utils import save_model
import datetime
import numpy as np

# -------------------------- Global settings

METRICS = {'accuracy': accuracy_score, 'f1': f1_score}
BASIC_MODELS = [
    ('Majority Class', MajorityClass),
    ('Logistic Regression', LogisticRegression)
    ]
NOFSPLITS = 5

# -------------------------- Public functions
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
    res = []
    res.extend(_eval_basic_models(data, METRICS))
    return res
  else:
    bm = _select_best_model(data["models"])
    yhat = bm.predict(data['X_test'])
    idx2label = {i: label for label, i in data["label2idx"].items()}
    res = [idx2label[i] for i in yhat]
    return res

# -------------------------- Private functions

def _select_best_model(models):
  """Returns the best model.

  Parameters
  ----------
  models : list
    List with dictionaries including info about given models.

  Returns
  -------
  bm : sklearn estimator
    Best model to use for the future predictions. 
  """
  
  scores = np.array([md["score"] for md in models])
  mds = np.array([md["model"] for md in models])
  bm = mds[np.argmax(scores)]

  return  bm

def _eval_basic_models(data, METRICS):

  """Evaluates basic models.

  Basic model is for example Logistic regression or SVM.
  Or in genera, anything which is not Neural network.

  Parameters
  ----------
  data : dict
    Contains all the needed data to (train) and evaluate given models.

  Returns
  -------
  res : list
    List where each item is a dictionary with corresponding info about
    the models' cross validated performance.
  """
  
  METRICS = {name: make_scorer(f) for name, f in METRICS.items()}

  res = []
  for name, bm in BASIC_MODELS:

    print(f"> {name} classifier (cross validated scores)")
    start = datetime.datetime.now()
    tmp = {'model_name': name}

    clf = bm(METRICS, NOFSPLITS)
    clf.fit(data["X_train"], data["y_train"])
    md = clf.model

    tmp["model"] = md
    tmp["score"] = md.best_score_
    print(f'>> F1 score: {tmp["score"]}')

    # save model
    save_model(md, 'data/models', name.lower().replace(' ', '_') + '.pkl', timestamp=False)

    end = datetime.datetime.now()
    diff = end - start
    tmp["training_time"] = diff
    print(f">> Running time: {diff}\n")

    res.append(tmp)
    
  return res

