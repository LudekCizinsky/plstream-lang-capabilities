# Import experiments for reproduction
from experiments import baseline_basic_testset, run_preprocessing_pipeline

# Define parsers
import argparse
parser = argparse.ArgumentParser(
    description='Control Runflow of main.py')

parser.add_argument(
    '-B1',
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

args = parser.parse_args()

def main():

  if args.baseline_eval_basic_testset:
    baseline_basic_testset()

  if args.basic_dataset_preprocess:
    run_preprocessing_pipeline()


if __name__ == "__main__":
  main()

