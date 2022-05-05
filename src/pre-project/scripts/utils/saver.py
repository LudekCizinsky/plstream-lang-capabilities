# saver.py
# script to store functions to save any kind of output
import os
import pickle
from datetime import datetime


def save_model(model, path, filename, timestamp=True):
  """Saved trained model into memory-efficient binary format
  using the pickle library. 

  Parameters
  ----------
  model : Object
    Trained ML model

  path : str
    Path to directory, where model should be saved to

  filename : str
    Filename of the model (has to include some suffix, by
    default .pkl)

  timestamp : bool
    Boolean to control whether or not to timestamp the
    filename of the model that is being saved

  Returns
  -------
  None
  """
  if timestamp:
    today = datetime.now().strftime('%Y-%m-%d-%H-%M')
    name, suffix = name.split('.')
    filepath = f"{path}/{filename}{today}{suffix}"
  else:
    filepath = f"{path}/{filename}"

  os.makedirs(path) if not os.path.exists(path) else None
  with open(filepath, "wb") as f:
    pickle.dump(model, f)
