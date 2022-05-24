import random
import copy
import re
import numpy as np
import sys
import pandas as pd
import glob
import os
from pathlib import Path
from scipy.special import softmax
from scipy.spatial import distance
from pyflink.datastream.connectors import StreamingFileSink
from pyflink.common.serialization import Encoder
from time import time

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec

import redis
import pickle
import logging

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode



class for_output(MapFunction):
  def __init__(self):
    pass

  def map(self, value):
    return str(value[1])


class unsupervised_OSA(MapFunction):

  def __init__(self):
    # collection
    self.true_label = []
    self.collector = []
    self.cleaned_text = []
    self.stop_words = stopwords.words('english')
    self.collector_size = 2000

    # model pruning
    self.LRU_index = ['good', 'bad']
    self.max_index = max(self.LRU_index)
    self.LRU_cache_size = 30000
    #     self.sno = nltk.stem.SnowballStemmer('english')

    # model merging
    self.flag = True
    self.model_to_train = None
    self.timer = time()
    import math
    self.time_to_reset = math.inf 

    # similarity-based classification preparation
    self.true_ref_neg = []
    self.true_ref_pos = []
    self.ref_pos = ['love', 'best', 'beautiful', 'great', 'cool', 'awesome', 'wonderful', 'brilliant', 'excellent',
            'fantastic']
    self.ref_neg = ['bad', 'worst', 'stupid', 'disappointing', 'terrible', 'rubbish', 'boring', 'awful',
            'unwatchable', 'awkward']
    # self.ref_pos = [self.sno.stem(x) for x in self.ref_pos]
    # self.ref_neg = [self.sno.stem(x) for x in self.ref_neg]

    # temporal trend detection
    self.pos_coefficient = 0.5
    self.neg_coefficient = 0.5

    # results
    self.confidence = 0.5
    self.acc_to_plot = []
    self.predictions = []

    self.labelled_dataset = ""

  def open(self, runtime_context: RuntimeContext):
    # redis-server parameters
    self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)

    # load initial model
    self.initial_model = Word2Vec.load('PLS_c10.model')
    self.vocabulary = list(self.initial_model.wv.index_to_key)

    # save model to redis
    self.save_model(self.initial_model)

  def save_model(self, model):
    print("Saving Model to Redis")
    self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    try:
      self.redis_param.set(f'osamodel_{TRAINING_SIZE}', pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except (redis.exceptions.RedisError, TypeError, Exception):
      logging.warning('Unable to save model to Redis server, please check your model')

  def load_model(self):
    self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    try:
      called_model = pickle.loads(self.redis_param.get(f'osamodel_{TRAINING_SIZE}'))
      return called_model
    except TypeError:
      logging.info('The model name you entered cannot be found in redis')
    except (redis.exceptions.RedisError, TypeError, Exception):
      logging.warning('Unable to call the model from Redis server, please check your model')

  # tweet preprocessing
  def text_to_word_list(self, text):
    text = text.replace("\n", "") # remove newline
    text = re.sub("@\w+ ", "", text)
    text = re.sub("[!~#$+%*:()'?-]", ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    clean_word_list = text.strip().split(' ')
    clean_word_list = [w for w in clean_word_list if w not in self.stop_words]
    while '' in clean_word_list:
      clean_word_list.remove('')
    self.cleaned_text.append(clean_word_list)
    if len(self.cleaned_text) >= self.collector_size:
      ans = self.update_model(self.cleaned_text)
      return ans
    else:
      return ('collecting', '1')

  def model_prune(self, model):

    if len(model.wv.index_to_key) <= self.LRU_cache_size:
      return model
    else:
      word_to_prune = list(self.LRU_index[30000:])
      for word in word_to_prune:
        k = model.wv.key_to_index[word]
        del model.wv.index_to_key[k]
        del model.wv.key_to_index[word]
      self.vocabulary = list(model.wv.index_to_key)
      return model

  def get_model_new(self, final_words, final_vectors, final_syn1, final_syn1neg, final_cum_table, corpus_count,
            final_count, final_sample_int, final_code, final_point, model):

    model_new = copy.deepcopy(model)
    n_words = len(final_words)
    model_new.wv.index_to_key = final_words
    model_new.wv.key_to_index = {word: idx for idx, word in enumerate(final_words)}
    model_new.wv.vectors = final_vectors
    model_new.syn1 = final_syn1
    model_new.syn1neg = final_syn1neg
    model_new.syn1 = final_syn1
    model_new.syn1neg = final_syn1neg
    model_new.cum_table = final_cum_table
    model_new.corpus_count = corpus_count
    model_new.corpus_total_words = n_words
    model_new.wv.expandos['count'] = final_count
    model_new.wv.expandos['sample_int'] = final_sample_int
    model_new.wv.expandos['code'] = final_code
    model_new.wv.expandos['point'] = final_point
    return model_new

  def model_merge(self, model1, model2):
    if model1[0] == 'labelled':
      return model2[1]
    elif model1[0] == 'model':
      model1 = model1[1]
      model2 = model2[1]
      words1 = copy.deepcopy(model1.wv.index_to_key)
      words2 = copy.deepcopy(model2.wv.index_to_key)
      syn1s1 = copy.deepcopy(model1.syn1)
      syn1s2 = copy.deepcopy(model2.syn1)
      syn1negs1 = copy.deepcopy(model1.syn1neg)
      syn1negs2 = copy.deepcopy(model2.syn1neg)
      cum_tables1 = copy.deepcopy(model1.cum_table)
      cum_tables2 = copy.deepcopy(model2.cum_table)
      corpus_count = copy.deepcopy(model1.corpus_count) + copy.deepcopy(model2.corpus_count)
      counts1 = copy.deepcopy(model1.wv.expandos['count'])
      counts2 = copy.deepcopy(model2.wv.expandos['count'])
      sample_ints1 = copy.deepcopy(model1.wv.expandos['sample_int'])
      sample_ints2 = copy.deepcopy(model2.wv.expandos['sample_int'])
      codes1 = copy.deepcopy(model1.wv.expandos['code'])
      codes2 = copy.deepcopy(model2.wv.expandos['code'])
      points1 = copy.deepcopy(model1.wv.expandos['point'])
      points2 = copy.deepcopy(model2.wv.expandos['point'])
      final_words = []
      final_vectors = []
      final_syn1 = []
      final_syn1neg = []
      final_cum_table = []
      final_count = []
      final_sample_int = []
      final_code = []
      final_point = []
      for idx1 in range(len(words1)):
        word = words1[idx1]
        v1 = model1.wv[word]
        syn11 = syn1s1[idx1]
        syn1neg1 = syn1negs1[idx1]
        cum_table1 = cum_tables1[idx1]
        count = counts1[idx1]
        sample_int = sample_ints1[idx1]
        code = codes1[idx1]
        point = points1[idx1]
        try:
          idx2 = words2.index(word)
          v2 = model2.wv[word]
          syn12 = syn1s2[idx2]
          syn1neg2 = syn1negs2[idx2]
          cum_table2 = cum_tables2[idx2]
          v = np.mean(np.array([v1, v2]), axis=0)
          syn1 = np.mean(np.array([syn11, syn12]), axis=0)
          syn1neg = np.mean(np.array([syn1neg1, syn1neg2]), axis=0)
          cum_table = np.mean(np.array([cum_table1, cum_table2]), axis=0)
        except:
          v = v1
          syn1 = syn11
          syn1neg = syn1neg1
          cum_table = cum_table1
        final_words.append(word)
        final_vectors.append(list(v))
        final_syn1.append(syn1)
        final_syn1neg.append(syn1neg)
        final_cum_table.append(cum_table)
        final_count.append(count)
        final_sample_int.append(sample_int)
        final_code.append(code)
        final_point.append(point)
      for idx2 in range(len(words2)):
        word = words2[idx2]
        if word in final_words:
          continue
        v2 = model2.wv[word]
        syn12 = syn1s2[idx2]
        syn1neg2 = syn1negs2[idx2]
        cum_table2 = cum_tables2[idx2]
        count = counts2[idx2]
        sample_int = sample_ints2[idx2]
        code = codes2[idx2]
        point = points2[idx2]
        try:
          idx1 = words1.index(word)
          v1 = model1.wv[word]
          syn11 = syn1s1[idx1]
          syn1neg1 = syn1negs1[idx1]
          cum_table1 = cum_tables1[idx1]
          v = np.mean(np.array([v1, v2]), axis=0)
          syn1 = np.mean(np.array([syn11, syn12]), axis=0)
          syn1neg = np.mean(np.array([syn1neg1, syn1neg2]), axis=0)
          cum_table = np.mean(np.array([cum_table1, cum_table2]), axis=0)
        except:
          v = v2
          syn1 = syn12
          syn1neg = syn1neg2
          cum_table = cum_table2
        final_words.append(word)
        final_vectors.append(list(v))
        final_syn1.append(syn1)
        final_syn1neg.append(syn1neg)
        final_cum_table.append(cum_table)
        final_count.append(count)
        final_sample_int.append(sample_int)
        final_code.append(code)
        final_point.append(point)

      model_new = self.get_model_new(final_words, np.array(final_vectors), np.array(final_syn1),
                       np.array(final_syn1neg), \
                       final_cum_table, corpus_count, np.array(final_count),
                       np.array(final_sample_int), \
                       np.array(final_code), np.array(final_point), model1)
      self.save_model(model_new)
      self.flag = True
      return model_new

  def map(self, tweet):
    self.true_label.append(int(tweet[1]))
    self.collector.append(tweet[0])

    return self.text_to_word_list(tweet[0])

  def update_model(self, new_sentences):

    if self.flag:
      call_model = self.load_model()
      self.flag = False
    else:
      call_model = self.model_to_train

    # incremental learning
    call_model.build_vocab(new_sentences, update=True)  # 1) update vocabulary
    print(len(call_model.wv.key_to_index))
    call_model.train(new_sentences,  # 2) incremental training
             total_examples=call_model.corpus_count,
             epochs=call_model.epochs)
    for word in call_model.wv.index_to_key:
      if word not in self.vocabulary:  # new words
        self.LRU_index.insert(0, word)
      else:  # duplicate words
        self.LRU_index.remove(word)
        self.LRU_index.insert(0, word)
    self.vocabulary = list(call_model.wv.index_to_key)
    self.model_to_train = call_model

    if len(self.ref_neg) > 0:
      for words in self.ref_neg:
        if words in call_model.wv:
          self.ref_neg.remove(words)
          if words not in self.true_ref_neg:
            self.true_ref_neg.append(words)
    if len(self.ref_pos) > 0:
      for words in self.ref_pos:
        if words in call_model.wv:
          self.ref_pos.remove(words)
          if words not in self.true_ref_pos:
            self.true_ref_pos.append(words)

    classify_result = self.eval(new_sentences, call_model)
    self.cleaned_text = []
    self.true_label = []

    if time() - self.timer >= self.time_to_reset:
      call_model = self.model_prune(call_model)
      model_to_merge = ('model', call_model)
      self.timer = time()
      return model_to_merge
    else:
      not_yet = ('labelled', classify_result)
      return not_yet

  def eval(self, tweets, model):
    self.labelled_dataset = ""
    for i in range(len(tweets)):
      tweet = tweets[i]
      preds = self.predict(tweet, model)
      label = preds[0]
      prob_neg = preds[1]
      prob_pos = preds[2]
      # prob_neu = preds[2]

      self.labelled_dataset += f"{self.collector[i]}\t{label}\t{prob_neg}\t0.0\t{prob_pos}\n"
      self.predictions.append(preds[0])

    self.collector = []

    # no trend detection
    # self.neg_coefficient = self.predictions.count(0) / self.predictions.count(1)
    # self.pos_coefficient = 1 - self.neg_coefficient

    return self.labelled_dataset

  def predict(self, tweet, model):
    sentence = np.zeros(20)
    counter = 0
    cos_sim_bad, cos_sim_good = 0, 0
    # print(tweet)
    for word in tweet:
      try:
        sentence += model.wv[word]  # np.array(list(model.wv[words]) + new_feature)
        counter += 1
      except:
        pass
    if counter != 0:
      sentence_vec = sentence / counter
      #print('got sentence vec')
    # else:
      #print('couldnt get sentence')

    k_cur = min(len(self.true_ref_neg), len(self.true_ref_pos))
    for neg_word in self.true_ref_neg[:k_cur]:
      try: 
        cos_sim = (1 - distance.cosine(model.wv[neg_word], sentence_vec))
        cos_sim_bad += cos_sim
      except: pass

    for pos_word in self.true_ref_pos[:k_cur]:
      try:
        cos_sim = (1 - distance.cosine(model.wv[pos_word], sentence_vec))
        cos_sim_good += cos_sim
      except: pass

    #print(cos_sim_bad, cos_sim_good)
    if cos_sim_bad == 0 and cos_sim_good == 0:
      # model cannot evaluate document because of
      # missing vocab
      print('getting at random')
      cos_sim_bad, cos_sim_good = np.random.uniform(0, 1, 2)
      s = cos_sim_bad + cos_sim_good
    else:
      s = cos_sim_bad + cos_sim_good

    cos_prob_bad, cos_prob_good = softmax([cos_sim_bad, cos_sim_good])
    print(cos_prob_bad, cos_prob_good)

    if cos_prob_bad > cos_prob_good:
      # print(0, cos_sim_bad / s, cos_sim_good / s)
      return (0, cos_prob_bad, cos_prob_good)
    else:
      # print(1, cos_sim_bad / s, cos_sim_good / s)
      return (1, cos_prob_bad, cos_prob_good)
        


def plstream(python_path, data_path, train=True):
  print('-- Getting stream ready')
  # parallelism = 4 # didnt get this to work yet

  # load train data
  df = pd.read_csv("./data/train.csv", index_col=False)

  # store reviews and label in lists
  true_label = list(df.label)
  yelp_review = list(df.review)

  # create data stream of list of tuples
  data_stream = []
  for i in range(len(yelp_review)):
    data_stream.append( (yelp_review[i], int(true_label[i])))

  # load checklist test in to list of tests
  with open("data/checklist-tests.txt", "r") as f:
    checklist_tests = [line.strip() for line in f]
  

  # separate training from checklist stream
  # training stream: [0, 560.000] (560.000)
  # checklist stream: [560.001, 647.470] (87.470)
  global TRAINING_SIZE
  TRAINING_SIZE = 10 * 10**4
  training_stream = data_stream[:TRAINING_SIZE]
  dummies = ["<this is a dummy>" for _ in range(len(checklist_tests)%2000)]
  checklist_stream = [(test, 0) for test in checklist_tests + dummies]

  print('> Done\n')
  
  if train:
    print('-- Setting up the job')
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_python_executable(python_path)
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    ds = env.from_collection(collection=training_stream+checklist_stream)
    s1 = ds.map(unsupervised_OSA()).filter(lambda x: x[0] != 'collecting')
    s2 = s1.key_by(lambda x: x[0], key_type=Types.STRING())
    s3 = s2.reduce(lambda x, y: (x[0], unsupervised_OSA().model_merge(x, y)))
    s4 = s3.filter(lambda x: x[0] != 'model').map(for_output(), output_type=Types.STRING())
    s5 = s4.add_sink(StreamingFileSink.for_row_format('./output', Encoder.simple_string_encoder()).build())
    print('> Done\n')

    print('-- Running the job on local cluster')
    env.execute("osa_job")

  print('-- Formatting Output')

  # getting most recent output file
  output_files = glob.glob('./output/*/.*', recursive=True)
  recent_files = sorted(output_files, key=os.path.getctime, reverse=True)

  cols = ["text", "prediction", "prob_neg", "prob_neu", "prob_pos"]

  data1 = pd.read_csv(recent_files[0], sep="\t", names=cols)
  data2 = pd.read_csv(recent_files[1], sep="\t", names=cols)

  print(f'> Loading most recent predictions from {recent_files[0]}')
  print(f'> Loading most recent predictions from {recent_files[1]}')
  data = pd.concat([data2, data1]).reset_index().drop('index', axis=1)
  data = data.iloc[-88000:-(88000-87470)]
  data["prediction"] = data["prediction"].astype(int)
  data["prediction"] = data["prediction"].map({0:0, 1:2})

  data = data[["prediction", "prob_neg", "prob_neu", "prob_pos"]]
  print('> Wrote Predictions to predictions/plstream')
  data.to_csv(f"predictions/plstream_{TRAINING_SIZE}", 
      sep=" ", 
      index=False, 
      header=False)

  print('Done!')
