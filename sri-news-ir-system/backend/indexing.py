import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

def build_tfidf_index(corpus):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(corpus["text"])
    return tfidf, matrix

def build_bm25_index(corpus):
    tokenized = [doc.split() for doc in corpus["text"]]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized
