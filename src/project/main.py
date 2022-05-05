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
SUITE_PATH = 'sentiment_suite.pkl'
PRED_PATH = 'predictions'
PYTHON_PATH = subprocess.check_output(
    "which python", shell=True).strip().decode('utf-8')

# parser
parser = argparse.ArgumentParser(
    description='Control Runflow of main.py')

parser.add_argument(
    '-R',
    '--reproduce-checklist',
    action='store_true',
      help='Reproduce the Checklist Paper Results')

parser.add_argument(
    '-F',
    '--format-plstream-train',
    action='store_true',
      help='Format PLStream Training Data and\
      write to data/final.csv')

args = parser.parse_args()

def main():
  total = timer() # global timer
  
  # reproduce checklist results
  if args.reproduce_checklist:
    s = working_on('Reproduce Checklist Results')
    suite = TestSuite.from_file(SUITE_PATH)

    models = os.listdir(PRED_PATH)

    for model in models:
      sys.stdout = open(f"summaries/{model}.txt", "wt")

      print(f"Summary from {model.upper()}")
      suite.run_from_file(f"{PRED_PATH}/{model}", overwrite=True)
      suite.summary()

    sys.stdout = sys.__stdout__
    finished('Reproduce Checklist Results', timer() - s)

  # checklist pl-stream
  if args.format_plstream_train:
    s = working_on('Format PLStream Training Data')

    # preprocess testing data into format of
    shutil.copyfile("data/train.csv", "data/final.csv")

    with open("data/final.csv", "a") as final:
      with open("data/checklist-tests.txt", "r") as tests:
        for test in tests:
          test = test.strip().replace('"', '')
          t = f'0,"{test}"\n'
          final.write(t)

    finished('Format PLStream Training Data', timer() - s)

  # train plstream on this data
  s = working_on('Checklist PLStream')
  
  plstream(PYTHON_PATH, data_path="data/final.csv")

  finished('Checklist PLStream', timer() - s)

  finished('Entire Pipeline', timer() - total)

if __name__ == "__main__":
  main()
