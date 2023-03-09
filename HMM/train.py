import os
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time


ROOT_DIR = '../' if 'HMM' in os.getcwd() else os.getcwd() # setting the root dir
POS_DIR = os.path.join(ROOT_DIR, 'dataset') # setting the pos dir

pos_train = os.path.join(POS_DIR, "train.txt") 

def format_data(fname):
    sentences = [] # master list
    with open(fname) as f:
        content = f.readlines()
    
    sentence = [] # local list
    for line in content:
        if line !='\n':
            line = line.strip() # remove leading/trailing spaces
            word = line.split()[0].lower() # get the word
            pos = ""
            pos = line.split()[1] # get the pos tag
            sentence.append((word, pos)) # create a pair and save to local list
        else:
            sentences.append(sentence) # once a \n is detected, append the local sentence to master sentence
            sentence = []
    return sentences

datalist = format_data(pos_train)

print(len(datalist))

train_set,test_set =train_test_split(datalist,train_size=0.80,test_size=0.20,random_state = 101)
print(len(train_set), len(test_set))

train_tagged_words = [ tup for sent in train_set for tup in sent ]
test_tagged_words = [ tup for sent in test_set for tup in sent ]
print(len(train_tagged_words), len(test_tagged_words))

tags = {tag for word,tag in train_tagged_words}
print(tags, len(tags))
vocab = {word for word,tag in train_tagged_words}


def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
#now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
 
     
    return (count_w_given_tag, count_tag)

def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
 
print(tags_matrix)