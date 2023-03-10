# Logistic Regression for POS tagging

# PACKAGES
import os


def dataset_init():
    # SET DATASET PATHS
    dataset_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset')).replace("\\", "/")
    train_data_path = "/train.txt"
    test_data_path = "/test.txt"
    # ==
    if not os.path.exists(dataset_path + train_data_path) or not os.path.exists(dataset_path + test_data_path):
        raise FileNotFoundError("Check dataset paths!")
    return dataset_path + train_data_path, dataset_path + test_data_path


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


if __name__ == "__main__":
    # Initialize Datasets
    train_path, test_path = dataset_init()

    # Preprocessing on training dataset
    train_sentences = format_data(train_path)

    print("Tagged sentences in train set: ", len(train_sentences))
    print("Tagged words in train set:", len([item for sublist in train_sentences for item in sublist]))

