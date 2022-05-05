# _logistic_regression.py
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------- Public
class LogisticRegression:

  def __init__(self, metrics, cv):
    self.metrics = metrics
    self.cv = cv
    self.model = None
 
  def fit(self, X, y):
    
    pipe = Pipeline([
      ('vectorize', CountVectorizer()),
      ('LR', LR(random_state=42))
    ])

    parameters = { 
        "LR__C": [0.3, 0.05], 
        "LR__max_iter": [500, 1000]
    }

    self.model = GridSearchCV(
        pipe,
        parameters,
        scoring=self.metrics,
        refit='f1',
        cv=self.cv,
        n_jobs=-1,
        verbose=3)

    
    self.model.fit(X, y)
  
  def predict(X):
    return self.model.predict(X)

