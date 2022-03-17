# main.py
from scripts import load_data
from scripts.baseline import MajorityClass, LogisticRegression, SOTA

def main():
  load_data()

if __name__ == "__main__":
  main()
