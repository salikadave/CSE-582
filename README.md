## Part-of-Speech Tagging

### Dataset

We use the training dataset obtained from [CoNLL-2000](https://www.cnts.ua.ac.be/conll2000/chunking/) and word embeddings from [wiki-news-300d-1M](https://fasttext.cc/docs/en/english-vectors.html). To download the datatset and embeddings, run [download.sh](dataset/download.sh)



## Folder and file structure

* Our implementation of machine learning models, 
    - Logisitic Regression (LR) can be found in [LR-notebook.ipynb](LR/LR-notebook.ipynb)
    - Multi-Layer Perceptron (MLP) can be found in [MLP.ipynb](MLP/MLP.ipynb)
    - Hidden Markov Models (HMM) can be found in  [HMM.ipynb](HMM/HMM.ipynb) 
    - Ensemble of the above models can be found in [ensemble_model.ipynb](ensemble_model.ipynb)
* [src](src/) folder has our common utils and functions to process data
* [linguisticlions.test.txt](linguisticlions.test.txt) is Labeled test data by our best/final model. Labelled data by other models can be found in [Labelled_outputs](Labelled_outputs/)