# default libraries
import re
import sys
import copy
import pickle
import random
import logging

import redis
import numpy as np
import pandas as pd
from time import time
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import accuracy_score

# word2vec model
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

# nlp
import nltk
from nltk.corpus import stopwords

# pyflink
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode

from pyflink.datastream.connectors import StreamingFileSink
from pyflink.common.serialization import Encoder

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class for_output(MapFunction):
  def __init__(self):
    pass

  def map(self, value):
    return str(value[1])

class PLStream(MapFunction):
  def __init__(self):
    print('init')
    # collection
    self.true_label = []
    self.collector = []
    self.cleaned_text = []
    self.collector_size = 5 

    # model pruning
    self.LRU_index = ['good','bad']
    self.max_index = max(self.LRU_index)
    self.LRU_cache_size = 30000 #Least Recently Used Cache discards the least recently used items first

    # preprocessing
    self.stop_words = stopwords.words('english')
    self.sno = nltk.stem.SnowballStemmer('english')

    # model merging
    self.flag = True
    self.model_to_train = None
    self.timer = time()
    self.time_to_reset = 5
    self.trained = False

    # similarity-based classification preparation
    self.true_ref_neg = []
    self.true_ref_pos = []
    self.ref_pos = ['love', 'best', 'beautiful', 'great', 'cool', 
        'awesome', 'wonderful', 'brilliant', 'excellent', 'fantastic']
    self.ref_neg = ['bad', 'worst', 'stupid', 'disappointing', 'terrible', 
        'rubbish', 'boring', 'awful', 'unwatchable', 'awkward']

    # temporal trend detection
    self.pos_coefficient = 0.5
    self.neg_coefficient = 0.5

    # results
    self.acc_to_plot = []
    self.predictions = []
    self.labelled_dataset = ''

  def open(self, runtime_context: RuntimeContext):
    print('open')
    # redis-server parameters 
    self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)

    # load initial model
    self.initial_model = Word2Vec.load('./scripts/models/word2vec.model')

    print(self.initial_model.wv.index_to_key)
    self.vocabulary = list(self.initial_model.wv.index_to_key)

    # save model to redis
    self.save_model(self.initial_model)

  def save_model(self, model):
    print('save_model')
    self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    try:
      self.redis_param.set('word2vec', pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except (redis.exceptions.RedisError, TypeError, Exception):
      logging.warning('Unable to save model to Redis server, please check your model')

  def load_model(self):
    print('load_model')
    self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    try:
      return pickle.loads(self.redis_param.get('word2vec'))
    except TypeError:
      logging.info('The model name you entered cannot be found in redis')
    except (redis.exceptions.RedisError, TypeError, Exception):
      logging.warning('Unable to call the model from Redis server, please check your model')

  # tweet preprocessing
  def text_to_word_list(self, text):
    print('text_to_word_list')
    # could use general purpose nltk tokeniser here (because not tweet data)
    text = re.sub("@\w+ ", "", text) # deletes user name from tweet
    text = re.sub("[!~#$+%*:()'?-]", ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    clean_word_list = text.strip().split(' ')
    clean_word_list = [w for w in clean_word_list
        if w not in self.stop_words and w != '']

    # append tokenised tweet onto collector
    self.cleaned_text.append(clean_word_list)
    
    # if exceeds stream size (collector size)
    # retrain word2vec model
    if len(self.cleaned_text) >= self.collector_size:
      ans = self.update_model(self.cleaned_text)
      return ans
    else:
      return ('collecting', '1')

  def model_prune(self, model):
    print('model_prune')
    if len(model.wv.index_to_key) <= self.LRU_cache_size:
      return model
    else:
      word_to_prune = list(self.LRU_index[30000:])
      # Here I think we delete the old words (we keep only the most recent 30000 words)
      for word in word_to_prune:
        k = model.wv.key_to_index[word]
        del model.wv.index_to_key[k]
        del model.wv.key_to_index[word]

      # new vocabulary has only the 30000 most recent words
      self.vocabulary = list(model.wv.index_to_key)
      return model

  def get_model_new(self, final_words, final_vectors, final_syn1, final_syn1neg, final_cum_table, corpus_count, final_count, final_sample_int, final_code, final_point, model):
    """
    Create a new model from the old one (a copy of
    a model) Is used when we want to merge two
    existing models
    """
    print('get_model_new')

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
    """
    Takes two models and merges them using
    get_model_new(), it basically checks for each
    parameter(word,its vector and other stuff) if
    it is in the other model, if it is, it does
    some statistic summary of those two models,
    otherwise it just gets the parameter that is
    in either of them
    """
    print('model_merge')
    if model1[0] == 'labelled':
      return (model1[1]) + (model2[1])
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
    print('map')
    self.true_label.append(int(tweet[1]))
    self.collector.append(tweet[0])
    return self.text_to_word_list(tweet[0])

  def update_model(self, new_sentences):
    print('update_model')
    if self.flag:
      call_model = self.load_model()
      self.flag = False
    else:
      call_model = self.model_to_train

    # incremental learning
    #print(set([item for sublist in
    #  new_sentences for item in sublist]))
    # unique words in this training batch

    # print(call_model.wv.index_to_key)
    call_model.build_vocab(new_sentences, update=True)  # 1) update vocabulary
    print(call_model.wv.index_to_key)
    call_model.train(new_sentences,  # 2) incremental training
             total_examples=call_model.corpus_count,
             epochs=call_model.epochs)

    for word in call_model.wv.index_to_key:
      # for a word in the models vocabulary
      if word not in self.vocabulary:  # new words
        self.LRU_index.insert(0, word)
      else:  # duplicate words
        self.LRU_index.remove(word)
        self.LRU_index.insert(0, word)

    self.vocabulary = list(call_model.wv.index_to_key)
    self.model_to_train = call_model

    # Create true_ref_neg and true_ref_pos based on the words that actually appear in the original ref_neg and ref_pos and the actual model
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

    # IF a certain time (30 units) has passed we prune the model, and return it so it can be merged
    if time() - self.timer >= self.time_to_reset:
      call_model = self.model_prune(call_model)
      model_to_merge = ('model', call_model)
      self.timer = time()
      return model_to_merge
    # If not then we just keep classifying the incoming sentence
    else:
      not_yet = ('labelled', classify_result)
      return not_yet

  def eval(self, tweets, model):
    print('eval')
    # print('VOCAB SIZE', len(model.wv.index_to_key))
    for i in range(len(tweets)):
      predict_result = self.predict(tweets[i], model)
      self.predictions.append(predict_result)
      self.labelled_dataset += predict_result+'\n'
    #self.neg_coefficient = self.predictions.count('0')/self.predictions.count('1')
    #self.pos_coefficient = 1 - self.neg_coefficient
    self.predictions = []
    self.collector = []
    ans = self.labelled_dataset
    return ans

  def predict(self, tweet, model):
    print('predict')
    sentence = np.zeros(20)
    counter = 0
    cos_sim_bad, cos_sim_good = 0, 0
    # Represent the tweet as a sum of its word vectors
    for words in tweet:
      try:
        sentence += model.wv[words]  # np.array(list(model.wv[words]) + new_feature)
        counter += 1
      except:
        pass
    if counter != 0:
      sentence_vec = sentence / counter
    k_cur = min(len(self.true_ref_neg), len(self.true_ref_pos))
    for neg_word in self.true_ref_neg[:k_cur]:
      try:
        cos_sim_bad += dot(sentence_vec, model.wv[neg_word]) / (norm(sentence_vec) * norm(model.wv[neg_word]))
      except:
        pass
    for pos_word in self.true_ref_pos[:k_cur]:
      try:
        cos_sim_good += dot(sentence_vec, model.wv[pos_word]) / (norm(sentence_vec) * norm(model.wv[pos_word]))
      except:
        pass
    if cos_sim_bad - cos_sim_good > 0.5:
      return '0'
    elif cos_sim_bad - cos_sim_good < -0.5:
      return '1'
    else:
      if cos_sim_bad * self.neg_coefficient >= cos_sim_good * self.pos_coefficient:
        return '0'
      else:
        return '1'

def plstream(python_path, data_path):
  print('-- Getting stream ready')

  # load data
  f = pd.read_csv(data_path, index_col=False)
  
  # subset of training data
  n = 20000
  true_label = list(f.label)#[:n]
  yelp_review = list(f.review)#[:n]

  # get training data stream ready
  data_stream = []
  for i in range(len(yelp_review)):
    data_stream.append((yelp_review[i], int(true_label[i])))
  print('> Done\n')
  
  print('-- Setting up the job')
  env = StreamExecutionEnvironment.get_execution_environment()
  env.set_parallelism(1) # must be present otherwise a grpc error occurs
  env.set_python_executable(python_path)
  env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

  training_stream = data_stream[:2000]
  checklist_stream = data_stream[-87470:]

  """
  PLStream().open(None) # reload initial word2vec with V=2
  params = redis.StrictRedis(host='localhost', port=6379, db=0)
  model =  pickle.loads(params.get('word2vec'))
  print(model.wv.key_to_index)
  """

  #model = PLStream()
  ds = env.from_collection(collection=training_stream)
  s1 = ds.map(PLStream()).filter(lambda x: x[0] != 'collecting')
  s2 = s1.key_by(lambda x: x[0], key_type=Types.STRING())
  s3 = s2.reduce(lambda x, y: (x[0], PLStream().model_merge(x, y)))
  s4 = s3.filter(lambda x: x[0] != 'model').map(for_output(), output_type=Types.STRING())
  s5 = s4.add_sink(StreamingFileSink.for_row_format('./output', Encoder.simple_string_encoder()).build())
  print('> Done\n')

 # ds.map(PLStream()) \
 #   .filter(lambda x: x[0] != 'collecting')\
 #   .key_by(lambda x: x[0], key_type=Types.STRING()) \
 #   .reduce(lambda x, y: (x[0], PLStream().model_merge(x, y))) \
 #   .filter(lambda x: x[0] != 'model') \
 #   .map(for_output(), output_type=Types.STRING()) \
 #   .add_sink(StreamingFileSink.for_row_format('./output', Encoder.simple_string_encoder()).build())

  # checklist stream
  print(f'Number of cores used: {env.get_parallelism()}')
  
  print('-- Running the job on local cluster')
  env.execute("osa_job")

  params = redis.StrictRedis(host='localhost', port=6379, db=0)
  model =  pickle.loads(params.get('word2vec'))
  print(model.wv.key_to_index)
  return
  predictor = PLStream()
  preds = predictor.eval([i[0] for i in checklist_stream], model)
  preds = preds.replace('1','2').replace('0','1')

  with open('predictions/plstream', 'w') as f:
    f.write(preds)
