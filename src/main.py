from scripts.feature_ext import get_training_data
from scripts.evaluation import evaluate_baseline
import warnings
warnings.filterwarnings('ignore')

def main():

  print("-------------- Loading training data")
  X_train, y_train = get_training_data() # train + dev
  print()

  print("-------------- (Training) and evaluating models")
  models = evaluate_baseline({"X_train": X_train, "y_train": y_train})

if __name__ == "__main__":
  main()
