# -*- coding: utf-8 -*-
"""
Created on Thu May 24 00:10:18 2018

@author: Pranjal
"""
import trbm
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from matplotlib import style
from nltk.tokenize import word_tokenize
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression


style.use("ggplot")
###########################################################################################################

### THIS BLOCK IS TO BE RUN ONLY ONCE

# Get the list of folders in bbc
# Each folder is a tag

folder_list = os.listdir("bbc")
folders = []
folders += [name for name in folder_list if not name.endswith(".TXT")]

complete_input = dict((key, []) for key in folders)
# {'business': [], 'entertainment': [], 'politics': [], 'sport': [], 'tech': []}

for x in folders:
    for file in os.listdir("bbc\\" + x):
        complete_input[x] += open("bbc\\" + x + "\\" + file).readlines()

### Now we have the complete input for each tag
        
###########################################################################################################

complete_input['business'] = [w for w in complete_input['business'] if not w in stopwords.words()]
complete_input['entertainment'] = [w for w in complete_input['entertainment'] if not w in stopwords.words()]
complete_input['politics'] = [w for w in complete_input['politics'] if not w in stopwords.words()]
complete_input['sport'] = [w for w in complete_input['sport'] if not w in stopwords.words()]
complete_input['tech'] = [w for w in complete_input['tech'] if not w in stopwords.words()]


###########################################################################################################

### TEXTRANK

# business
summary_defb, summary_30b, summary_40b, summary_50b = trbm.textrank(complete_input['business'], stopwords=stopwords)

# ententainment
summary_defe, summary_30e, summary_40e, summary_50e = trbm.textrank(complete_input['entertainment'], stopwords=stopwords)

# politics
summary_defp, summary_30p, summary_40p, summary_50p = trbm.textrank(complete_input['politics'], stopwords=stopwords)

# sport
summary_defs, summary_30s, summary_40s, summary_50s = trbm.textrank(complete_input['sport'], stopwords=stopwords)

# tech
summary_deft, summary_30t, summary_40t, summary_50t = trbm.textrank(complete_input['tech'], stopwords=stopwords)


### BM25

# business
bm25out_10b, bm25out_30b, bm25out_40b, bm25out_50b = trbm.bm25(complete_input['business'])

# ententainment
bm25out_10e, bm25out_30e, bm25out_40e, bm25out_50e = trbm.bm25(complete_input['entertainment'])

# politics
bm25out_10p, bm25out_30p, bm25out_40p, bm25out_50p = trbm.bm25(complete_input['politics'])

# sport
bm25out_10s, bm25out_30s, bm25out_40s, bm25out_50s = trbm.bm25(complete_input['sport'])

# tech
bm25out_10t, bm25out_30t, bm25out_40t, bm25out_50t = trbm.bm25(complete_input['tech'])



###########################################################################################################

### SENTIMENT ANALYSIS 

## Vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

## comparing politics

# 30%
s30t = sid.polarity_scores(''.join(summary_30p))
s30b = sid.polarity_scores(''.join(bm25out_30p))

# 40%
s40t = sid.polarity_scores(''.join(summary_40p))
s40b = sid.polarity_scores(''.join(bm25out_40p))

# 50%
s50t = sid.polarity_scores(''.join(summary_50p))
s50b = sid.polarity_scores(''.join(bm25out_50p))

## Logistic Regression

pos_vocab = word_tokenize(open("short_reviews/positive.txt","r").read())
neg_vocab = word_tokenize(open("short_reviews/negative.txt", "r").read())

pos_vocab = [w for w in pos_vocab if not w in stopwords.words("english")]
neg_vocab = [w for w in neg_vocab if not w in stopwords.words("english")]

def word_feats(words):
    return dict([(word, True) for word in words])


def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

positive_features = [(word_feats(pos), 'pos') for pos in pos_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in neg_vocab]

train_set = negative_features + positive_features

LRclassifier = SklearnClassifier(LogisticRegression())
LRclassifier.train(train_set)



def pre(text):
    text = word_tokenize(''.join(text).lower())
    neg = 0
    pos = 0
    for word in text:
        classResult = LRclassifier.classify( word_feats(word))
        if classResult == 'neg':
            neg = neg + 1
        if classResult == 'pos':
            pos = pos + 1
        outdict = {'pos': str(float(pos)/len(text)), 'neg' : str(float(neg)/len(text))}
    return outdict

# 30%
ss30t = pre(summary_30p)
ss30b = pre(bm25out_30p)

# 40%
ss40t = pre(summary_40p)
ss40b = pre(bm25out_40p)

# 50%
ss50t = pre(summary_50p)
ss50b = pre(bm25out_50p)


###########################################################################################################

## DataFrame for the final output

index_list = [['Vader', 'Logistic Regression'], ['TextRank', 'BM25']]

index = pd.MultiIndex.from_product(index_list, names=['Classifier', 'Summarization'])
final = pd.DataFrame(index=index, columns=['30%', '40%', '50%'])

final.loc[('Vader', 'TextRank')] = [s30t, s40t, s50t]
final.loc[('Vader', 'BM25')] = [s30b, s40b, s50b]

final.loc[('Logistic Regression', 'TextRank')] = [ss30t, ss40t, ss50t]
final.loc[('Logistic Regression', 'BM25')] = [ss30b, ss40b, ss50b]


###########################################################################################################