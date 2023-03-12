
import os
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time
from tqdm import tqdm
import pickle
import tables


ROOT_DIR = '../' if 'HMM' in os.getcwd() else os.getcwd() # setting the root dir
POS_DIR = os.path.join(ROOT_DIR, 'dataset') # setting the pos dir

pos_train = os.path.join(POS_DIR, "train.txt") 
pos_test = os.path.join(POS_DIR, "test.txt") 


patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'AT'),   # articles
    (r'.*able$', 'JJ'),                # adjectives
    (r'.*ness$', 'NN'),                # nouns formed from adjectives
    (r'.*ly$', 'RB'),                  # adverbs
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # past tense verbs
    (r'.*', 'NN'),                     # nouns
]
 
# rule based tagger
rule_based_tagger = nltk.RegexpTagger(patterns)

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

def compute_emmision(word, tag, train_bag= train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
     
    return count_w_given_tag /count_tag


def compute_transition(t2, t1):
    count_t1 = len([t for t in transition_tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(transition_tags)-1):
        if transition_tags[index]==t1 and transition_tags[index+1] == t2:
            count_t2_t1 += 1
    return count_t2_t1/count_t1


def test_accuracy(algorithm, tagged, untagged):
    start = time.time()
    tagged_seq = algorithm(untagged)
    end = time.time()
    difference = end-start
    
    print("Time taken in seconds: ", difference)
    
    # accuracy
    check = [i for i, j in zip(tagged_seq, tagged) if i == j] 
    
    accuracy = len(check)/len(tagged_seq)
    print('Viterbi Algorithm Accuracy: ',accuracy*100)
    return tagged_seq, check, accuracy


def Viterbi_rule_based(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    ROW_SIZE = len(tags)
    filename = 'probabilities.h5'
    with tables.open_file(filename, mode='w') as f:
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0, ROW_SIZE))
    
     
    for key, word in tqdm(enumerate(words), total=len(words)):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                 
            # compute emission and state probabilities
            emission_p = compute_emmision(words[key], tag)
            state_probability = emission_p * transition_p    
            p.append(state_probability)
             
        pmax = max(p)
        state_max = rule_based_tagger.tag([word])[0][1]       
        
         
        if(pmax==0):
            state_max = rule_based_tagger.tag([word])[0][1] # assign based on rule based tagger
        else:
            if state_max != 'X':
                # getting state for which probability is maximum
                state_max = T[p.index(pmax)]  
        p = np.array(p).reshape(1, ROW_SIZE)  
        with tables.open_file(filename, mode='a') as f:
            f.root.data.append(p)               
             
        state.append(state_max)
    return list(zip(words, state))

def read_probs():
    with tables.open_file('probabilities.h5', mode='r') as f:
        data = f.root.data[:]
    return data


def main():
    
    data = format_data(pos_train)
    
    global train_set
    global test_set
    train_set, test_set = train_test_split(data, train_size=.8, test_size=.20, random_state=42)

    global train_tagged_words
    train_tagged_words = [ tup for sent in train_set for tup in sent ]

    test_tagged_words = [tup for sent in test_set for tup in sent]

    test_untagged_words =  [word for sent in test_set for word in sent]


    global tags
    tags = {tag for word,tag in train_tagged_words}
    global vocab
    vocab = {word for word,tag in train_tagged_words}

    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)): 
            tags_matrix[i, j] = compute_transition(t2, t1)

    global tags_df
    tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
    global transition_tags
    transition_tags = [pair[1] for pair in train_tagged_words]


    tagged_seq, check, accuracy = test_accuracy(Viterbi_rule_based, test_tagged_words, test_untagged_words)

    with open('output.pkl', 'wb') as f:
        pickle.dump([tagged_seq, check, accuracy], f)
        

if __name__ == "__main__":
    main()