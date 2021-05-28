import random
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import re
import os
from sklearn.utils import shuffle
from math import log
from nltk.corpus import stopwords
import nltk
import pandas as pd
import requests
import pickle as pkl
import json
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from pprint import pprint
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import string
nltk.download('stopwords') 
nltk.download('wordnet') 
nltk.download('punkt')


def black_txt(token, stop_words_, common_terms):  
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2  
 
def clean_txt(text, stop_words_, common_terms):  
    clean_text = [] 
    clean_text2 = []
    #print(text)
    #text = stem_text(text)  
    clean_text = [ word for word in nltk.word_tokenize(text.lower()) if black_txt(word, stop_words_, common_terms)]  
    clean_text2 = [word for word in clean_text if black_txt(word, stop_words_, common_terms)]  
    return " ".join(clean_text2)
 
def preprocess(corpus):  
    stop = open('terrier-stop.txt','r')  
    stopString = stop.read()  
    common_terms = stopString.split()  
    stop_words_ = set(nltk.corpus.stopwords.words('english'))  
    #wn = WordNetLemmatizer()  
    temp_corpus = nltk.word_tokenize(clean_txt(corpus, stop_words_, common_terms))
    return temp_corpus

def train_ngram(corpus): 
    bigram = Phrases(corpus, min_count=2, threshold=0.65) 
    bcorpus = bigram[corpus] 
    Model_bg = Phraser(bigram) 
    trigram = Phrases(bcorpus, min_count = 2, threshold = 0.65) 
    Model_tg = Phraser(trigram)
    return Model_bg, Model_tg 

def extract_ngrams(BModel, TModel, corpus): 
    ngram = TModel[BModel[corpus]]
    return ngram 

def remove_emoji(string):
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

## Conceptnet parsing 
def extract_triples(tokens, intID):
    Sub = []
    rel = []
    Obj = []
    Id = []
    for i in tokens:
        link = 'http://api.conceptnet.io/c/en/'+ i
        obj = requests.get(link).json()
        top = dict()
        if len(obj['edges']) == 0:
            obj = requests.get("http://api.conceptnet.io/related/c/en/"+i).json()
            for j in obj['related']:
                if (float(j['weight'])>=0.5 and j['@id'].count("/en/")>0):
                    temp = set()
                    link = 'http://api.conceptnet.io/c/en/'+ j['@id'].split("/en/")[1]
                    obj = requests.get(link).json()
                    for edge in obj['edges']:
                        try:
                            if(edge['end']['language']=='en'):
                                # temp.add((i,edge['rel']['label'], edge['end']['label']))
                                Id.append(intID)
                                Sub.append(i)
                                Obj.append(edge['end']['label'])
                                rel.append(edge['rel']['label'])
                        except KeyError:
                            continue
                if(len(temp)!=0):
                    top[j['@id'].split("/en/")[1]] = list(temp)
                else:
                    continue
        else:
            temp = set()
            for edge in obj['edges']:
                try:
                    if(edge['end']['language']=='en'):
                        # temp.add((i,edge['rel']['label'], edge['end']['label']))
                        Id.append(intID)
                        Sub.append(i)
                        Obj.append(edge['end']['label'])
                        rel.append(edge['rel']['label'])
                except KeyError:
                    continue
    return Id, Sub, rel, Obj

data = [json.loads(l) for l in open('ALONE-Toxicity-Dataset_v5_1.json', 'r')]
df = pd.DataFrame(data)

# print(df[1])

corpus = []
ngrams_interaction = {}
for i in range(len(df['Tweets'])):
  Tweets = []
  for tweet in df['Tweets'][i]:
    Tweets.append(remove_emoji(''.join(c if c not in map(str,range(0,10)) else "" for c in tweet)))
    corpus.append(" ".join(Tweets))
  inter = preprocess(corpus[i])
  MB, TB = train_ngram(inter)
  ngrams = extract_ngrams(MB, TB, inter)
  ngrams_interaction[i] = ngrams


print("---- Triples ----")

triples = []
MId = []
MSub = []
Mrel = []
MObj = []

for i in range(len(ngrams_interaction)):
    print("---Extract---")
    Id, Sub, rel, Obj = extract_triples(ngrams_interaction[i], i)
    MId = MId + Id
    MSub = MSub + Sub
    Mrel = Mrel + rel
    MObj = MObj + Obj

triples = pd.DataFrame(list(zip(MId, MSub, Mrel, MObj)))
triples.to_csv("triples.csv")
