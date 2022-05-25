## About 

The goal of the pre-project phase was to
build a baseline model (in our case `Logistic
regression`) and then tested it on:

- [basic test
  set](data/raw/music_reviews_test_masked.json.gz)
  provided by our course manager
- [difficult
  cases](data/difficult_cases/phase2_testData-masked.json.gz)
  created by the entire class in order to
  challenge each other's baseline models


## Reproducing Results 

### Baseline Model Training and Prediction on the basic test set 

To reproduce predictions obtained by our baseline
model on the basic test set, run:

```bash 
python main.py -BAS 
```

You can then find the results in the [basic
testset
predictions](results/basic_testset_predictions)
folder. In addition, if you are interested how we
[preprocessed data](data/processed/), you can run
the following preprocessing pipeline that
reproduces the extraction part:

```bash 
python main.py -PRE 
```

Finally, below is a short description of our
baseline model:
- As our baseline model, we ended up using
  'Logistic regression'
- The system for Logistic Regression was designed
  as a short pipeline, where the raw reviews are
  transformed using 'CountVectorizer' with its
  default settings which represents each review as
  BOW
- The best set of hyper-parameters (different
  number of iterations and different strenghts of
  regularization) was found using 'GridSearch'
  from sklearn. The best model achieved cross
  validated (5 folds) F1 score: 0.94

### Creating custom difficult cases and baseline model evaluation 

First, to see how we have
produced [difficult
cases](results/custom_difficult_cases) using
[checklist](https://github.com/marcotcr/checklist)
framework and how our baseline model performed on
them, run:

```bash 
python main.py -HT 
```

We have described the way we came up with these
test cases
[here](https://docs.google.com/presentation/d/1skA61kkbNvFxbgi2Y1uIfML8z7xb1l8MMnpNEuRxVGI/edit?usp=sharing).
One of the starting points for us was to examine
what are the
[mispredictions](results/mispredictions/) of our
model. To reproduce the mispredictions, you can
run the following:

``` 
python main.py -MIS 
``` 

Finally,
[predictions](results/difficult_testset_predictions)
of our baseline model on the [difficult
cases](data/difficult_cases/phase2_testData-masked.json.gz)
can be reproduced by running:

```bash 
python main.py -DIF 
```
