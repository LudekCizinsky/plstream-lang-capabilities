# _sota.py
import pandas as pd
import math
import gzip
import json
from pytorch_transformers import XLNetTokenizer
from transformers import XLNetForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
from pytorch_transformers import AdamW
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class SOTA():
  def __init__(self):
    self.model = None
    self.data = None
    self.max = None
    self.BATCH = 3
    self.LR = 2e-5
    self.EPOCHS = 3
    self.LIMIT = (True,10) #Cuts sentences so that the maximum length is 10, otherwise it runs forever

  def process(self,x):
    """
    Takes data from load_data defined in loader.py

    x = load_data(split='train')
    model.process(x)
    
    """
    data = {'sentence':[], 'value':[]}
    for i in range(len(x)):
      try:
        data['sentence'].append(x[i]['reviewText'])
        if x[i]['sentiment'] == 'positive':
          data['value'].append(1)
        
        elif x[i]['sentiment'] == 'negative':
          data['value'].append(0)
        
      except:
        pass

    self.data = pd.DataFrame.from_dict(data)

    sentences  = []
    for sentence in self.data['sentence']:
      sentence = sentence+"[SEP] [CLS]"
      sentences.append(sentence)

    
    tokenizer  = XLNetTokenizer.from_pretrained('xlnet-base-cased',do_lower_case=True)
    tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]

    idxs = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

    if self.LIMIT[0]:
      idxs = [idx[:self.LIMIT[1]] for idx in idxs]
    
    self.findmax(idxs)

    #Pad the sentences
    input_idxs = pad_sequences(idxs,maxlen=self.max,dtype="long",truncating="post",padding="post")
    labels = self.data['value'].values

    X = torch.tensor(input_idxs)
    Y = torch.tensor(labels)

    train_data = TensorDataset(X,Y)
    loader = DataLoader(train_data,batch_size=self.BATCH)

    return loader

  def fit(self,x):
    """
    Fit data loaded from a loader defined in loader.py
    X = load_data(split='train')
    """
    
    loader = self.process(x)
    print("Data succesfully processed")
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",num_labels=2)
    print("XLnet successfully loaded")
    optimizer = AdamW(model.parameters(),lr=self.LR)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no_train = 0
    print('Starting training')
    for epoch in range(self.EPOCHS):
      model.train()
      loss1 = []
      steps = 0
      train_loss = []
      l = []
      for inputs,labels1 in tqdm(loader):
        inputs.to(device)
        labels1.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        loss = criterion(outputs[0],labels1.to(device)).to(device)
        logits = outputs[0]
        [train_loss.append(p.item()) for p in torch.argmax(outputs[0],axis=1).flatten() ]#our predicted 
        [l.append(z.item()) for z in labels1]# real labels
        loss.backward()
        optimizer.step()
        loss1.append(loss.item())
        no_train += inputs.size(0)
        steps += 1
      print("Current Loss is : {} Step is : {} number of Example : {} Accuracy : {}".format(loss.item(),epoch,no_train,self.accuracy(train_loss,l)))


    self.model = model

  def findmax(self,idxs):
    max1 = len(idxs[0])
    for i in idxs:
      if(len(i)>max1):
        max1=len(i)
    self.max = max1

  def accuracy(self,preds,labels):
    correct=0
    for i in range(0,len(labels)):
      if(preds[i]==labels[i]):
        correct+=1
    return (correct/len(labels))*100

  def predict(self,x):
    loader = self.process(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.model.eval()
    acc = []
    lab = []
    t = 0
    for inp,lab1 in loader:
      inp.to(device)
      lab1.to(device)
      t+=lab1.size(0)
      outp1 = self.model(inp.to(device))
      [acc.append(p1.item()) for p1 in torch.argmax(outp1[0],axis=1).flatten() ]
      [lab.append(z1.item()) for z1 in lab1]
    print("Total Examples : {} Accuracy {}".format(t,self.accuracy(acc,lab)))

    return acc

  
if __name__ == '__main__':
  def load_data(split='dev', subsample=math.inf):
    """
    Loading specified raw data split as a list of dictionaries.
    Each dictionary represents a single review.
    """
    if split not in ['train', 'dev', 'test']:
      raise ValueError("split not defined. choose from ['train', 'dev', 'test']")
    if split == 'test':
      split = 'test_masked'

    data = []
    for i, line in enumerate(gzip.open(f'../../data/classification/music_reviews_{split}.json.gz')):
      review_data = json.loads(line)
      data.append(review_data)
      
      if i >= subsample:
        break

    return data

  x= load_data('train')
  model = SOTA()
  #model.process(x)
  model.fit(x)
  #print(model.data)
