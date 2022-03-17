# main.py
from scripts import load_data
from scripts.baseline import MajorityClass, LogisticRegression, SOTA

def main():
  train_data = load_data(split='train')
  dev_data = load_data(split='dev')
  test_data = load_data(split='test')

if __name__ == "__main__":
  main()
