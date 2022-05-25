#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# custom modules
from scripts.models import plstream
from scripts.utils import output, working_on, finished

# default modules
import os
import sys
import json
import shutil
import argparse
import datetime
import subprocess
from timeit import default_timer as timer

# external modules
import checklist
from checklist.test_suite import TestSuite

# global envs
SUITE_PATH = 'data/checklist_testsuite/sentiment_suite.pkl'
PRED_PATH = 'results/checklist_predictions'
PYTHON_PATH = subprocess.check_output(
    "which python3", shell=True).strip().decode('utf-8')

# parser
parser = argparse.ArgumentParser(
    description='Control Runflow of main.py')

parser.add_argument(
    '-T',
    '--train-plstream',
    action='store_true',
      help='Train PLStream')

parser.add_argument(
    '-R',
    '--reproduce-checklist',
    action='store_true',
      help='Reproduce the Checklist Paper Results')

args = parser.parse_args()

def main():
  total = timer() # global timer

  # train plstream on this data
  if args.train_plstream:
    s = working_on('Training PLStream')
    plstream(PYTHON_PATH,
        data_path="data/processed/final.csv", 
        train=args.train_plstream)
    finished('Checklist PLStream', timer() - s)

  # reproduce checklist results
  if args.reproduce_checklist:
    s = working_on('Reproduce Checklist Results')
    suite = TestSuite.from_file(SUITE_PATH)

    models = os.listdir(PRED_PATH)

    for model in models:
      sys.stdout = open(f"results/checklist_summaries/{model}", "wt")
      print(f"Summary of {model.upper()}")

      print(f"{PRED_PATH}/{model}")
      suite.run_from_file(f"{PRED_PATH}/{model}", overwrite=True)
      suite.summary()
      sys.stdout = sys.__stdout__

    finished('Reproduce Checklist Results', timer() - s)

  finished('Entire Pipeline done', timer() - total)

if __name__ == "__main__":
  main()

