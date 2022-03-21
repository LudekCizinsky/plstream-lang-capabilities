#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("-------------- Loading modules")
from scripts.feature_ext import get_training_data, get_test_data
from scripts.evaluation import evaluate_baseline
import datetime
import os
print("Done!\n")


def main():

  print("-------------- Loading training data")
  X_train, y_train, label2idx  = get_training_data() # train + dev
  print("Done!\n")

  print("-------------- (Training) and evaluating models")
  models = evaluate_baseline({"X_train": X_train, "y_train": y_train})
  print("Done!\n")

  print("-------------- Loading test data")
  X_test  = get_test_data(label2idx)
  print("Done!\n")

  print("-------------- Choosing best model and making predictions")
  yhat = evaluate_baseline(
      {"X_test": X_test, 'models': models, 'label2idx': label2idx},
      training=False
  )
  file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
  path = f"data/predictions/{file_name}.yhat"
  with open(path, "w") as f:
    res = "\n".join(yhat) 
    f.write(res)
  print(f"Predictions saved to: {path}")
  print("Done!\n")


if __name__ == "__main__":
  main()
