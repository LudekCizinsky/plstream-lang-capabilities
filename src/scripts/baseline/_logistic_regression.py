# _logistic_regression.py
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV

class LogisticRegression:

  def __init__(self):
    self.model = None

  def fit(self, X, y):
    
    parameters = { 
        "tol": [.001, .0001, .00001],
        "C": [0.3, 0.05],
        "solver": ['newton-cg', 'saga'],
        "max_iter": [100, 1000]
    }

    clf = GridSearchCV(LR(random_state=42), parameters)
    clf.fit(X, y) 

    self.model = clf.best_estimator_

  def predict(self, X):
    
    return self.model.predict(X)

