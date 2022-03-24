# Schedule

Task chosen: Binary Classification of Movie Reviews

Pipeline:
- [x] data loading 
- [x] preprocessing data (Mika)
  - [x] extract relevant information from json fields
  - [ ] save untokenised preprocessed data
  - [x] tokenise (nltk.word_tokenize)
  - [x] bow (sklearn.OneHotEncoder)

- [x] Baseline Models (24.03)
  - [x] Baseline 1: Majority Class Classifier (Iben)
  - [x] Baseline 2: Simple ML Model + BOW (Ludek)
  - [ ] Baseline 2: Simple ML Model + word2vec embeddings 
        (Ludek)
  - [x] Baseline 3: SotA (Lukas)

Project flow
- [x] 1. Put the baseline prediction on CodaLab(23.03.)
- [x] 2. Put the description of the baseline on Learnit (.txt file with 200 words) (23.03.)
- [ ] 3. Hand in difficult cases (30.03.)
- [ ] 4. Predict difficult cases using our baseline (06.04.)
- [ ] 5. Presentation of proposals (05.04 - 07.04.)
- [ ] 5. Upload our proposal on LearnIT (21.04.)
- [ ] 6. Upload draft (not mandatory) (19.05.)
- [ ] 7. Final project upload (27.05.)

1./2. Baseline Predictions/ Description

3. Difficult Cases
- [ ] extract wrong predictions
      - analyse manually
      - statistics false positives/ false negatives rate

- [ ] look at checklist repo
  (https://github.com/marcotcr/checklist#table-of-contents)
  and see if we can automatically generate difficult casees

- [ ] hand engineer difficult cases
      - negation
      - changed named entities
      - 

- [ ] 

4. Predictions on difficult cases

Structure:
- main.py - control flow of entire pipeline

Rob's Project Proposals
-----------------
1. Cross-domain (incorporate more data, reviews from other
   platform)
2. Improve on difficult cases without using them as training
   data
3. Low-resource model (How can we build a performant model
   in significantly less training time by using less
   training data)
4. Cross-lingual analysis (How can we train on language
   X and obtain good performance on language Y?) 

Ideas:
- explore different model architectures (compare sotas)
- bow, word2vec, elmo contextualised -> simple ML models
- huge, generalised pretrained models vs. smaller, specialised
  self-trained models
- sswe (sentiment specific word embeddings) - How can word
  vectors specifically learned for sentiment analysis tasks
  improve performance? (motivation: regular/ general
  embeddings do not encode similar sentiment, ie. good/ bad
  might have close embeddings)
- cross-lingual models 
- low resource model through feature extraction (but how can
  we do that in a low resource fashion)

  goal: reduce training and prediction time on classifier (low
        resource model)

  proposed solution: 
  different ways of feature extraction before feedings data
  into classifier:
  - naive way: use first/ last n tokens to feed through
    classifier
  - use pos tagger: use specific tags (found from a pretrained
    POS tagger) to feed through classifier

  always check for performance loss/ time gained balance.
