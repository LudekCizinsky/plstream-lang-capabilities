#_predict_difficult_cases.py
# script to save our predictions on the difficult cases defined by other students

from scripts.utils import load_model, get_difficult_cases, get_encodings
from scripts.utils import output, working_on, finished 
from timeit import default_timer as timer
import json

def main():
  total = timer() # global timer
  start = working_on('Loading Baseline Model')
  
  # Load the model
  model = load_model('data/models/logistic_regression.pkl')

  start = working_on('Getting the difficult cases')

  # Get the raw data so it can be saved in the same format and get the sentences for prediction
  raw = get_difficult_cases(which='raw')
  sentences = get_difficult_cases(which='sentences')

  # testing models
  start = working_on('Predicting on Difficult Cases')

  idx2label = get_encodings(['idx2label'])
  pred = model.predict(sentences)
  pred_decoded = [idx2label[str(label)] for label in pred]

  finished('Predicted on Difficult Cases', timer() - start)
  
  # save test predictions
  start = working_on('Saving Difficult Cases Predictions in Original Format')
  for r, y in zip(raw, pred_decoded):
    r["sentiment"] = y    
  file_name = 'TO_BE_RENAMED_FOR_CODALAB' 
  path = f"data/difficult_cases/{file_name}.json"
  with open(path, "w") as f:
    tmp = [json.dumps(r) for r in raw]
    res = "\n".join(tmp) 
    f.write(res)
  print(f"Predictions saved to: {path}")
  finished('Saving Difficult Cases in Original Format', timer() - start)

  finished('Entire Pipeline', timer() - total)

if __name__ == "__main__":
  main()
