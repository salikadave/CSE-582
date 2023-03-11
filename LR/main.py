# Logistic Regression for POS tagging

# PACKAGES
import os, json, gc, datetime

import numpy as np
import pickle
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from scipy.sparse import hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, confusion_matrix, f1_score
import nltk

embeddings_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset/embeddings')).replace("\\", "/")

def dataset_init():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset')).replace("\\", "/")
    train_data_path = "/train.txt"
    test_data_path = "/test.txt"
    labelled_test_data_path = "./test_labelled.txt"
    # ==
    if not os.path.exists(dataset_path + train_data_path) or not os.path.exists(dataset_path + test_data_path) or not\
            os.path.exists(dataset_path + labelled_test_data_path):
        raise FileNotFoundError("Check dataset paths!")
    return dataset_path + train_data_path, dataset_path + test_data_path, dataset_path + labelled_test_data_path


# RE-USE function from MLP for data preprocessing
def embeddings_init():
    filename = '/wiki-news-300d-1M.vec'
    # embeddings = {'data': 'sao'}
    embeddings = KeyedVectors.load_word2vec_format(embeddings_path + filename, binary=False)
    # save embeddings to a file
    # with open(embeddings_path + '/embeddings.json', 'w') as f:
    #     f.write(json.dumps({'data': embeddings}))
    return embeddings

# cannot store embeddings in a file -- redundant function
def load_embeddings_from_file(filepath):
    with open(filepath, 'r') as f:
        embeddings = f.read()
    return json.loads(embeddings)

# RE-USE function from MLP for data preprocessing
def format_data(fname, include_y=True):
    sentences = []  # master list
    with open(fname) as f:
        content = f.readlines()
    
    sentence = [] # local list
    for line in content:
        if line !='\n':
            line = line.strip() # remove leading/trailing spaces
            word = line.split()[0].lower()  # get the word
            if include_y:
                pos = ""
                pos = line.split()[1]  # get the pos tag
                sentence.append((word, pos))  # create a pair and save to local list
            else:
                sentence.append(word)
        else:
            sentences.append(sentence)  # once a \n is detected, append the local sentence to master sentence
            sentence = []
    return sentences

# RE-USE functions from MLP for word tags
def tag_sequence(sentences):
    return [[t for w, t in sentence] for sentence in sentences]

def text_sequence(sentences):
    return [[w for w, t in sentence] for sentence in sentences]

def id2word(sentences):
    wordlist = [item for sublist in text_sequence(sentences) for item in sublist]
    id2word = {k:v for k,v in enumerate(wordlist)}
    return id2word

def untag(tagged_sentence):
    return [w for w, _ in tagged_sentence]

def untag_pos(tagged_sentence):
    return [t for _, t in tagged_sentence]

def build_vocab(sentences):
    vocab =set()
    for sentence in sentences:
        for word in untag(sentence):
            vocab.add(word)
    return sorted(list(vocab))


# RE-USE function from MLP
def features_embs(sentence, index, window=1):
    # uses method concat
    unknown=0
    vec = np.array([])
    for i in range(index-window, index+window+1):
#         if i < 0:
#             vec = np.append(vec, pad)
#         if i > len(sentence)-1:
#             vec = np.append(vec, pad)
        try:
            vec = np.append(vec, embeddings[sentence[i]])
        except:
            vec = np.append(vec, oov)
            unknown += 1
    return vec, unknown


# RE-USE function from MLP
def features_basic(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'nb_terms': len(sentence),
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'i-1_prefix-3': '' if index == 0 else sentence[index-1][:3],
        'i-1_suffix-3': '' if index == 0 else sentence[index-1][-3:],
        'i+1_prefix-3': '' if index == len(sentence) - 1 else sentence[index+1][:3],
        'i+1_suffix-3': '' if index == len(sentence) - 1 else sentence[index+1][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
    }


# Re-use MLP
def transform_to_dataset(tagged_sentences, window):
    i=0
    X, y = [], []
    for doc_index, tagged in enumerate(tagged_sentences):
        for index in range(len(tagged)):
            X.append([features_basic(untag(tagged), index),\
                      features_embs(untag(tagged), index, window)[0],\
                     ])
            y.append(tagged[index][1])
            #features_embs(untag(tagged), index, window)[1]
            k = features_embs(untag(tagged), index, window)[1]
            i += k
    return X, y, i


# Re-use MLP
def transform_test_sentence(sentence, window):
    X = []
    for index in range(len(sentence)):
            X.append([
                      features_basic(sentence, index),\
                      features_embs(sentence, index, window),\
                     ])
    return X


# Re-use MLP
def transform_to_dataset_unknown(sentences, window):
    X = []
    for doc_index, sentence in enumerate(sentences):
        for index in range(len(sentence)):
            X.append([features_basic(sentence, index),\
                      features_embs(sentence, index, window),\
                     ])
    return X


# Re-use & modify MLP
def vectorize(train, window=1):
    # ===================================
    # using embeddings window method
    print('Embeddings window method')
    print('Vectorizing Dataset...')
    print('Vectorizing train...')
    X_train, y_train, unk_tr = transform_to_dataset(train, window=window)
    X_train = [x[1] for x in X_train]
    X_train = np.asarray(X_train)

    # ===================================
    # using boosted (window + classical)
    # print('Combined Classical - Embeddings window method')
    # print('Vectorizing Dataset...')
    # print('Vectorizing train...')
    # X_train, y_train, unk_tr = transform_to_dataset(train, window=window)
    # v = DictVectorizer(sparse=True)  # We choose sparse=True for faster concatenation later
    # X_classical = v.fit_transform([x[0] for x in X_train])
    # X_embs = [x[1] for x in X_train]
    # X_embs = np.asarray(X_embs)
    # X_train = hstack((X_classical, X_embs))
    # del X_classical, X_embs

    print('Dataset vectorized.')
    print('Train shape:', X_train.shape)
    return X_train, y_train


# Initialize Embeddings
t_ini = datetime.datetime.now()
print('Initializing embeddings')
embeddings = embeddings_init()
print('Initialiation completed')
t_fin = datetime.datetime.now()
print('Embeddings loaded in {} seconds'.format((t_fin - t_ini).total_seconds()))

# Global constants for feature engineering
dim = embeddings.vectors.shape[1]
pad = np.zeros(dim)  # Pad vector
np.random.seed(3)  # For reproducibility
oov = np.random.uniform(-0.25, 0.25, dim)  # Out-of-vocabulary vector


def init_training_without_cross_validation(X_train, y_train, model_filename='lr-model1.pkl'):
    t_ini = datetime.datetime.now()
    print('Training...')
    clf = LogisticRegression(C=1, solver='liblinear', multi_class='auto', random_state=2)
    clf.fit(X_train, y_train)
    t_fin = datetime.datetime.now()
    print('Training completed in {} seconds'.format((t_fin - t_ini).total_seconds()))

    # save model
    print('Saving model...')
    with open(model_filename, 'wb') as file:
        pickle.dump(clf, file)


def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        Pickled_LR_Model = pickle.load(file)

    return Pickled_LR_Model


def embed_test_sentences(sentence):
    X_embs = [x[1][0] for x in sentence]
    X_embs = np.asarray(X_embs)
    return X_embs


def preprocess_unlabelled_test_data(test_sentences):
    for sentence in test_sentences:
        sentence = transform_test_sentence(sentence, 1)
        embedded = embed_test_sentences(sentence)
        preprocessed_test_data.append(embedded)

    return preprocessed_test_data


if __name__ == "__main__":
    # Initialize Datasets
    train_path, test_path, labelled_test_path  = dataset_init()

    # Preprocessing on training dataset
    train_sentences = format_data(train_path)
    #
    print("Tagged sentences in train set: ", len(train_sentences))
    print("Tagged words in train set:", len([item for sublist in train_sentences for item in sublist]))
    #
    t_ini = datetime.datetime.now()
    print('Initializing vectorization...')
    X_train, y_train = vectorize(train_sentences, window=1)
    print('Completed vectorization...')
    t_fin = datetime.datetime.now()
    print('Vectorization completed in {} seconds'.format((t_fin - t_ini).total_seconds()))
    #
    init_training_without_cross_validation(X_train, y_train, model_filename='lr-model3.pkl')

    # Model Evaluation

    # Load Model
    clf = load_model(model_filename='lr-model3.pkl')

    # test_sentence = nltk.word_tokenize(
    #     'Word embeddings provide a dense representation of words and their relative meanings.'.lower())
    #
    # X_test_sentence = transform_test_sentence(test_sentence, window=1)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    correct_test_sen = format_data(labelled_test_path)
    test_sentences = format_data(test_path, False)
    preprocessed_test_data = []
    def embed_test_sentences(sentence):
        X_embs = [x[1][0] for x in sentence]
        X_embs = np.asarray(X_embs)
        return X_embs

    def preprocess_unlabelled_test_data(test_sentences):
        for sentence in test_sentences:
            sentence = transform_test_sentence(sentence, 1)
            embedded = embed_test_sentences(sentence)
            preprocessed_test_data.append(embedded)


    preprocess_unlabelled_test_data(test_sentences)

    predicted_data = []
    arg_max_dict = []


    def test_set_predictions(preprocessed_test_data, test_sentences):
        for sentence in preprocessed_test_data:
            predict_x = clf.predict(sentence)
            # predict_x = np.argmax(predict_x, axis=0)
            arg_max_dict.append(predict_x)

        for index in range(len(test_sentences)):
            predicted_sen = list(zip(test_sentences[index], arg_max_dict[index]))
            predicted_data.append(predicted_sen)


    test_set_predictions(preprocessed_test_data, test_sentences)


    def compare_with_test_set(correct_set):
        total = 0
        correct = 0
        for predicted_sentence, correct_sentence in zip(predicted_data, correct_set):
            for predicted_word, correct_word in zip(predicted_sentence, correct_sentence):
                total = total + 1
                if predicted_word[1] == correct_word[1]:
                    correct = correct + 1

        accuracy = (correct / total) * 100
        return accuracy

    print(compare_with_test_set(correct_test_sen))  # Accuracy = 94
    print(predicted_data)

    with open('output.txt', 'w') as f:
        f.write(str(predicted_data))

