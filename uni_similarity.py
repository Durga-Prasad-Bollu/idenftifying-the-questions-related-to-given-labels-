# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:26:51 2020

@author: bollud
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as inline
import nltk
import re

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

print("TF Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("TF Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

#==============================================================================
# tensroflow hub module for Universal sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" 
 #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

embed = hub.Module(module_url)
#==============================================================================

# load text
filename = (r"C:\Users\bollud\Desktop\links\link1.txt")
file = open(filename,'r')
text1 = file.read()
file.close()

filename2 = (r"C:\Users\bollud\Desktop\links\link2.txt")
file = open(filename2,'r')
text2 = file.read()
file.close()

filename3 = (r"C:\Users\bollud\Desktop\links\link3.txt")
file = open(filename3,'r')
text3 = file.read()
file.close()

filename4 = (r"C:\Users\bollud\Desktop\links\link4.txt")
file = open(filename4,'r')
text4 = file.read()
file.close()

filename5 = (r"C:\Users\bollud\Desktop\links\link5.txt")
file = open(filename5,'r')
text5 = file.read()
file.close()

#text1="The boy is playing cricket"
#text2="Ramesh is studing"
#text3="kiran is running to eat"
#text4="boys are very curious to know about cricket score"
#text5="Girls are playing cricket"

#==============================================================================
#TEXT-CLEANING
#==============================================================================

def text_lowercase(text): 
    return text.lower() 
#lower=text_lowercase(text1)

# Remove numbers 
def remove_numbers(text): 
    result = re.sub(r'\d+', '', text) 
    return result 
#removing_numbers=remove_numbers(lower) 

# remove punctuation 
import string
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator)
#punctuation=remove_punctuation(removing_numbers)
    

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
# remove stopwords function 
def remove_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return filtered_text 

#stop_words= " " .join(remove_stopwords(punctuation))

# remove whitespace from text 
def remove_whitespace(text): 
    return  " ".join(text.split()) 

#whitespace=remove_whitespace(stop_words) 
from nltk.tokenize import sent_tokenize, word_tokenize
def splitsent(text):
    sentences =nltk.sent_tokenize(text)
    no_of_sentences=len(text)
    return sentences
#no_of_sentences=splitsent(article_text)

def pre_process(text):
    lower=text_lowercase(text)
    removing_numbers=remove_numbers(lower) 
    punctuation=remove_punctuation(removing_numbers)
    stop_words= " " .join(remove_stopwords(punctuation))
    whitespace=remove_whitespace(stop_words) 
    no_of_sentences=splitsent(text)
    return whitespace

data=pre_process(text1)
data2=pre_process(text2)
data3=pre_process(text3)
data4=pre_process(text4)
data5=pre_process(text5)
#getting features for text data

#==============================================================================
# EXTRACTING EMBEDDINGS FOR SENTENCES
#==============================================================================

def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess.run(embed(texts))

BASE_VECTORS = get_features(data)
BASE_VECTORS2=get_features(data2)
BASE_VECTORS3=get_features(data3)
BASE_VECTORS4=get_features(data4)
BASE_VECTORS5=get_features(data5)
BASE_VECTORS3.shape
#==============================================================================


#def cosine_similarity(v1, v2):
#    mag1 = np.linalg.norm(v1)
#    mag2 = np.linalg.norm(v2)
#    if (not mag1) or (not mag2):
#        return 0
#    return np.dot(v1, v2) / (mag1 * mag2)
##
##def test_similarity(text1, text2):
##    vec1 = get_features(text1)[0]
##    vec2 = get_features(text2)[0]
##    print(vec1.shape)
##    print(vec2.shape)
##    return cosine_similarity(vec1, vec2)   
##
##test_similarity('Machine learning is a subfield of artificial intelligence (AI).','clarification needed Semi-supervised learning algorithms develop mathematical models from incomplete training data, where a portion of the sample input doesnt have labels.')
#
#
##data_processed =(list(map(data, text1))
#
#def test_similarity(text1, text2):
#    vec1 = get_features(text1)[0]
#    vec2 = get_features(text2)[0]
#    print(vec1.shape)
#    print(vec2.shape)
#    return cosine_similarity(vec1, vec2)   
#
#test_similarity('Machine learning is a subfield of artificial intelligence (AI).','Machine learning is a subfield of artificial intelligence (AI).')

#==============================================================================
#SPATIAL COSINE DISTANCE
#==============================================================================

from scipy import spatial

dataSetI =  [BASE_VECTORS]
dataSetII = [BASE_VECTORS2]
dataSetIII =[BASE_VECTORS3]
dataSetIV = [BASE_VECTORS4]
dataSetV =  [BASE_VECTORS5]
output3 = 1 - spatial.distance.cosine(dataSetI, dataSetII)

#==============================================================================

list1=[dataSetI,dataSetII,dataSetIII,dataSetIV,dataSetV]
list2=[dataSetI,dataSetII,dataSetIII,dataSetIV,dataSetV]

for i in range(len(list)):
    for j in range(len(list)):
        output4 = 1 - spatial.distance.cosine(list1, list2)
        
        
 

#def semantic_search(query, data, vectors):
#    query =data(query)
#    print("Extracting features...")
#    query_vec = get_features(query)[0].ravel()
#    res = []
#    for i, d in enumerate(data):
#        qvec = vectors[i].ravel()
#        sim = cosine_similarity(query_vec, qvec)
#        res.append((sim, d[:100], i))
#    return sorted(res, key=lambda x : x[0], reverse=True)
#
##semantic_search("machine learning", , BASE_VECTORS)
