
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
ROOT_DIR = '../' if 'HMM' in os.getcwd() else os.getcwd() # setting the root dir
POS_DIR = os.path.join(ROOT_DIR, 'dataset') # setting the pos dir

pos_train = os.path.join(POS_DIR, "train.txt") 
pos_test = os.path.join(POS_DIR, "test.txt") 


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

train_set = format_data(pos_train)
test_set = format_data(pos_test)
# train_set,test_set =train_test_split(datalist,train_size=0.80,test_size=0.20,random_state = 101)
print(len(train_set))
print(len(test_set))


# convert to dataframe
train_grouped = [ ["Sentence: " + str(sent_num+1), tup[0], tup[1]] for sent_num, sent in enumerate(train_set) for tup in sent ]
train_grouped
data_train = pd.DataFrame(train_grouped, columns=['sentence', 'Word', 'POS'])
data_train


test_grouped = [ ["Sentence: " + str(sent_num+1), tup[0], tup[1]] for sent_num, sent in enumerate(test_set) for tup in sent ]
data_test = pd.DataFrame(test_grouped, columns=['sentence', 'Word', 'POS'])
data_test


tags = list(set(data_train.POS.values)) 
train_vocab = list(set(data_train.Word.values))
data = pd.concat((data_test, data_train))
vocab = list(set(data.Word.values))
print(len(train_vocab), len(vocab))
# Convert words and tags into numbers
word2id = {w: i for i, w in enumerate(vocab)}
tag2id = {t: i for i, t in enumerate(tags)}
id2tag = {i: t for i, t in enumerate(tags)}


count_tags = dict(data_train.POS.value_counts())  # gets value counts for POS of the whole dataset
# gets a dictionary of the POS and all the value counts for each word for that POS
count_tags_to_words = data_train.groupby(['POS']).apply(lambda grp: grp.groupby('Word')['POS'].count().to_dict()).to_dict() 
# gets the POS value counts for all of the first values in a sentence
count_init_tags = dict(data_train.groupby('sentence').first().POS.value_counts())

# get number of times a certain tag prevtag, is followed by nexttag 
# this is for transmission matrix
count_tags_to_next_tags = np.zeros((len(tags), len(tags)), dtype=int)
sentences = list(data_train.sentence) # these are just a list of the sentence column
pos = list(data_train.POS) # these are just a list of the pos column
for i in range(len(data_train)) : # iterate through all words
    if (i > 0) and (sentences[i] == sentences[i - 1]): # if it is still the same sentence
        prevtagid = tag2id[pos[i - 1]]
        nexttagid = tag2id[pos[i]]
        count_tags_to_next_tags[prevtagid][nexttagid] += 1 
count_tags_to_next_tags[0]




mystartprob = np.zeros((len(tags),)) # probability it is a certain tag
mytransmat = np.zeros((len(tags), len(tags))) # transition matrix
myemissionprob = np.zeros((len(tags), len(vocab)))
num_sentences = sum(count_init_tags.values()) # used for probability of a certain tag
sum_tags_to_next_tags = np.sum(count_tags_to_next_tags, axis=1)
for tag, tagid in tag2id.items():
    floatCountTag = float(count_tags.get(tag, 0)) # number of tags in dataset
    mystartprob[tagid] = count_init_tags.get(tag, 0) / num_sentences
    for word, wordid in word2id.items():
        myemissionprob[tagid][wordid]= count_tags_to_words.get(tag, {}).get(word, 0) / floatCountTag
    for tag2, tagid2 in tag2id.items():
        mytransmat[tagid][tagid2]= count_tags_to_next_tags[tagid][tagid2] / sum_tags_to_next_tags[tagid]


model = hmm.MultinomialHMM(n_components=len(tags), algorithm='viterbi', random_state=42)
model.startprob_ = mystartprob
model.transmat_ = mytransmat
model.emissionprob_ = myemissionprob


samples = []
word_test = list(data_test.Word)
for i, val in enumerate(word_test):
    samples.append([word2id[val]])


lengths = []
count = 0
sentences = list(data_test.sentence)
for i in range(len(sentences)):
    if (i > 0) and (sentences[i] == sentences[i - 1]):
        count += 1
    elif i > 0:
        lengths.append(count)
        count = 1
    else:
        count = 1

lengths.append(count)

print(len(samples))
print(sum(lengths))
print(sum(lengths)==len(samples), sum(lengths)==len(data_test))
from sklearn.utils import check_array
X = check_array(samples)
from hmmlearn import _utils
for sub_X in _utils.split_X_lengths(X, lengths):
    print(sub_X.shape)


pos_predict = model.predict(samples, lengths)
pos_predict


