import numpy as np

from .feature_engineering import features_basic, features_embs
from .utils import untag

def transform_to_dataset(embeddings, oov, tagged_sentences, window):
    i=0
    X, y = [], []
    for doc_index, tagged in enumerate(tagged_sentences):
        for index in range(len(tagged)):
            X.append([features_basic(untag(tagged), index),\
                      features_embs(embeddings, oov, untag(tagged), index, window)[0],\
                     ])
            y.append(tagged[index][1])
            #features_embs(untag(tagged), index, window)[1]
            k = features_embs(embeddings, oov, untag(tagged), index, window)[1]
            i += k
    return X, y, i

def transform_test_sentence(embeddings, oov, sentence, window):
    X = []
    for index in range(len(sentence)):
            X.append([
                      features_basic(sentence, index),\
                      features_embs(embeddings, oov, sentence, index, window),\
                     ])
    return X

def transform_to_dataset_unknown(embeddings, oov, sentences, window):
    X = []
    for doc_index, sentence in enumerate(sentences):
        for index in range(len(sentence)):
            X.append([features_basic(sentence, index),\
                      features_embs(embeddings, oov, sentence, index, window),\
                     ])
    return X


def vectorize(embeddings, oov, train, window=1):
    # ===================================
    # using embeddings window method
    print('Embeddings window method')
    print('Vectorizing Dataset...')
    print('Vectorizing train...')
    X_train, y_train, unk_tr = transform_to_dataset(embeddings, oov, train, window=window)
    X_train = [x[1] for x in X_train]
    X_train = np.asarray(X_train)

    print('Dataset vectorized.')
    print('Train shape:', X_train.shape)
    return X_train, y_train


def embed_test_sentences(sentence):
    X_embs = [x[1][0] for x in sentence]
    X_embs = np.asarray(X_embs)
    return X_embs

def preprocess_unlabelled_test_data(embeddings, oov, test_sentences, window=1):

    preprocessed_test_data = []
    for sentence in test_sentences:
        sentence = transform_test_sentence(embeddings, oov, sentence, window=window)
        embedded = embed_test_sentences(sentence)
        preprocessed_test_data.append(embedded)

    return preprocessed_test_data

