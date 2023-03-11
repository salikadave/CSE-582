import numpy as np 

def features_embs(embeddings, oov, sentence, index, window=1):
    # uses method concat
    unknown=0
    vec = np.array([])
    for i in range(index-window, index+window+1):
        # if i < 0:
        #     vec = np.append(vec, pad)
        # if i > len(sentence)-1:
        #     vec = np.append(vec, pad)
        try:
            vec = np.append(vec, embeddings[sentence[i]])
        except:
            vec = np.append(vec, oov)
            unknown += 1
    return vec, unknown


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