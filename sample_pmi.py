import pandas as pd
import numpy as np
import math
import random
import string
import scipy.sparse as sp
from sklearn.utils import shuffle
from math import log
from nltk.corpus import stopwords
import nltk
import re
import requests
import json
import multiprocessing
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

import cosinesim
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords') 
nltk.download('wordnet') 
nltk.download('punkt')

model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
stop_words = stopwords.words('english')

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

data = [json.loads(l) for l in open('data/ALONE-Toxicity-Dataset_v5_1.json', 'r')]
df = pd.DataFrame(data)

print(df.shape)

corpus = []
for i in range(len(df['Tweets'])):
  Tweets = []
  for tweet in df['Tweets'][i]:
    Tweets.append(remove_emoji(''.join(c if c not in map(str,range(0,10)) else "" for c in tweet)))
    corpus.append(" ".join(Tweets))

def freq_with_cosine(text,fd1, word, threshold=0.2):
    """If cosine similarity meets the threshold, then increase frequence distribution
    Args:
        text ([string]): the whole document
        fdist ([type]): [description]
        word ([string]): candidate word
        threshold (float, optional): [description]. Defaults to 0.07.
    Returns:
        [int]: Updated frequency of the word
    """
    occurence = 1
    if word in stop_words:
        return occurence

    for w1 in fd1:
        #similarity = cosinesim.get_similarity_tfidf(model, stop_words, text, w1, word) 
        if w1 in stop_words:
            similarity = 0
        else:
            similarity = cosinesim.get_similartiy_bert(model, text, w1, word) #text: My school gave me an apple /w1: Apple/ word: Comptuer / compare Computer with text (with pretrained bert model)
        if similarity>threshold:
            occurence += 1
    return occurence 

striples = pd.read_csv("data/filtered_sample.csv")

# print(striples['1'][0])

def pmi(word1, word2, fd1, words, words1, words2, use_cosine):
    # word1 = porter.stem(word1)
    # word2 = porter.stem(word2)

    word_freq_1 = fd1[word1] if fd1[word1]>0 else 1
    word_freq_2 = fd1[word2] if fd1[word2]>0 else 1
    print(f"Freq: {word_freq_1}, {word_freq_2} ")
    if use_cosine:
        word_freq_1 += freq_with_cosine(corpus[27].lower(),fd1, word1)
        word_freq_2 += freq_with_cosine(corpus[27].lower(),fd1, word2)
    print(f"Freq after Cosine: {word_freq_1}, {word_freq_2} ")
    prob_word1 = word_freq_1/len(words1)
    prob_word2 = word_freq_2/len(words2)
    print(f"Prob: {prob_word1}, {prob_word2}")
    prob_word1_word2 = word_freq_1 * word_freq_2 / len(words)
    print(f"ProbW1W2: {prob_word1_word2}")
    pmi = math.log(prob_word1_word2/float(prob_word1*prob_word2), 2.71828)  # 0 for independence, and +1 for complete co-occurrence
    print(f"PMI : {pmi}")
    a = 1/len(words)
    b = 1 - a * len(words)
    npmi = a * pmi + b
    #npmi = pmi/log(1.0 * count / num_window) -1
    print(f"NPMI: {npmi:.4f}  ==> {word1}, {word2}")

    if npmi> 0.01:
        return True
    return False

words1 = list(striples['1'])
print(len(words1))
words2 = list(striples['3'])
print(len(words2))
words = words1 + words2

fd_one = nltk.FreqDist(words1) 
# fd_two = nltk.FreqDist(words2)

Final_triples = []

for i in range(len(striples)):
    #To try different words, change the "i" values in the line below
    if (i==50 or i==100 or i==300) and pmi(striples['1'][i], striples['3'][i], fd_one, words, words1, words2, True):
        Final_triples.append((striples['1'][i], striples['2'][i], striples['3'][i]))
    else:       
        # striples.drop(i, inplace = False)
        continue

# Final_triples = striples.reset_index(drop=True)

Final_triples = pd.DataFrame(Final_triples)

Final_triples.to_csv("CosinePmi.csv")
