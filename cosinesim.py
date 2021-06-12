# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sentence_transformers import  util

def get_similartiy_bert(model,text,w1,w2):
    embeddings1 = model.encode([text], convert_to_tensor=True)
    embeddings2 = model.encode([w1], convert_to_tensor=True)
    embeddings3 = model.encode([w2], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1 + embeddings2, embeddings3)
    return float(cosine_scores[0][0])
def get_similarity_tfidf(model, stop_words,text,w1,w2):

    def preprocess(sentence):
        return [w for w in sentence.lower().split() if w not in stop_words]

    sentence_1 = preprocess(text)
    sentence_2 = preprocess(w1)
    sentence_3 = preprocess(w2)

    from gensim.corpora import Dictionary
    documents = [sentence_1, sentence_2, sentence_3]
    #documents = [sentence_1, sentence_2]
    dictionary = Dictionary(documents)
    sentence_1 = dictionary.doc2bow(sentence_1)
    sentence_2 = dictionary.doc2bow(sentence_2)
    sentence_3 = dictionary.doc2bow(sentence_3)

    from gensim.models import TfidfModel
    documents = [sentence_1, sentence_2, sentence_3]
    #documents = [sentence_1, sentence_2]
    tfidf = TfidfModel(documents)
    sentence_1 = tfidf[sentence_1]
    sentence_2 = tfidf[sentence_2]
    sentence_3 = tfidf[sentence_3]
    
    from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
    termsim_index = WordEmbeddingSimilarityIndex(model)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    similarity = termsim_matrix.inner_product(sentence_1, sentence_2, normalized=(True, True))
    return similarity