# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:10:51 2018

@author: Pranjal
"""

# =============================================================================
# TEXTRANK
# =============================================================================

import numpy as np
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from operator import itemgetter


# PageRank Algorithm
def pagerank(A, esp=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= esp:
            return new_P
        P = new_P


# Function for calculating similaruty based on cosine
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
        
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
        
    return 1 - cosine_distance(vector1, vector2)


# Similarity matrix formation
def build_similarity_matrix(sentences, stopwords=None):
    # Start with an initial zero matrix
    S = np.zeros((len(sentences), len(sentences)))
    
    # filling the matrix
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords.words())
            
    # normalization
    for idx in range(len(S)):
        S[idx] /= S[idx].sum()
        
    return S


# final wrapper function
def textrank(sentences, top_n=5, stopwords=None):
    """
    sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
    top_n = how may sentences the summary should contain
    stopwords = a list of stopwords
    """
    S = build_similarity_matrix(sentences, stopwords)
    sentence_ranks = pagerank(S)
    
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences_def = sorted(ranked_sentence_indexes[:top_n])
    summary_def = itemgetter(*selected_sentences_def)(sentences)
    
    # 30%
    top_30 = int(0.3*len(sentences))
    selected_sentences_30 = sorted(ranked_sentence_indexes[:top_30])
    summary_30 = itemgetter(*selected_sentences_30)(sentences)
    
    # 40%
    top_40 = int(0.4*len(sentences))
    selected_sentences_40 = sorted(ranked_sentence_indexes[:top_40])
    summary_40 = itemgetter(*selected_sentences_40)(sentences)
    
    # 50%
    top_50 = int(0.5*len(sentences))
    selected_sentences_50 = sorted(ranked_sentence_indexes[:top_50])
    summary_50 = itemgetter(*selected_sentences_50)(sentences)
    
    return summary_def, summary_30, summary_40, summary_50


# =============================================================================
# BM25
# =============================================================================

from gensim.summarization.summarizer import summarize

def bm25(input_seq):
    inp = ''.join(input_seq)
    ten = summarize(inp, ratio=0.1, split=True)
    thirty = summarize(inp, ratio=0.3, split=True)
    forty = summarize(inp, ratio=0.4, split=True)
    fifty = summarize(inp, ratio=0.5, split=True)
    
    return ten, thirty, forty, fifty
    
# ============================================================================