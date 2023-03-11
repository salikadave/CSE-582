from gensim.models.keyedvectors import KeyedVectors

def format_data(fname, include_y=True):
    """
    Parses training data as sentences
    """
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


def embeddings_init(embeddings_path):
    """
    Loads word embeddings from downloaded file 
    """
    filename = '/wiki-news-300d-1M.vec'
    # embeddings = {'data': 'sao'}
    embeddings = KeyedVectors.load_word2vec_format(embeddings_path + filename, binary=False)
    # save embeddings to a file
    # with open(embeddings_path + '/embeddings.json', 'w') as f:
    #     f.write(json.dumps({'data': embeddings}))
    return embeddings