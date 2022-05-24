# Import experiments for reproduction
from experiments import baseline_basic_testset, run_preprocessing_pipeline, create_hard_tests, identify_mispredictions, predict_difficult_cases

# Define parsers
import argparse
parser = argparse.ArgumentParser(
    description='Control Runflow of main.py')

parser.add_argument(
    '-BAS',
    '--baseline_eval_basic_testset',
    action='store_true',
    help='Run best baseline model on the basic test cases.'
)

parser.add_argument(
    '-PRE',
    '--basic_dataset_preprocess',
    action='store_true',
    help='Run the preprocessing pipeline for basic dataset.'
)

parser.add_argument(
    '-HT',
    '--create_hard_tests',
    action='store_true',
    help='Creates hard tests and evaluates our baseline on it.'
)

parser.add_argument(
    '-L',
    '--load',
    action='store_true',
    help='load difficult cases otherwise, generate them'
)

parser.add_argument(
    '-MIS',
    '--identify_mispredictions',
    action='store_true',
    help='load difficult cases otherwise, generate them'
)

parser.add_argument(
    '-DIF',
    '--predict_diff_cases',
    action='store_true',
    help='Predict on difficult cases created by students.'
)

args = parser.parse_args()

def main():

  if args.baseline_eval_basic_testset:
    baseline_basic_testset()

  if args.basic_dataset_preprocess:
    run_preprocessing_pipeline()

  if args.create_hard_tests:
    create_hard_tests(args)

  if args.identify_mispredictions:
    identify_mispredictions()

  if args.predict_diff_cases:
    predict_difficult_cases()

if __name__ == "__main__":
  main()

