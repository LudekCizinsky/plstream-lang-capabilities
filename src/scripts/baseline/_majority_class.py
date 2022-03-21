import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV

# -------------------------- Public

class MajorityClass:

  def __init__(self, metrics, cv):
    self.metrics = metrics
    self.cv = cv
    self.model = None
 
  def fit(self, X, y):
    
    parameters = { 
        "strategy": ["prior"]
    }

    self.model = GridSearchCV(
        DummyClassifier(),
        parameters,
        scoring=self.metrics,
        refit='f1',
        cv=self.cv,
        n_jobs=-1,
        verbose=3)
 
    self.model.fit(X, y)
  
  def predict(X):
    return self.model.predict(X)

