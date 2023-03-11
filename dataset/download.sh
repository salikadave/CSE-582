if [ ! -e train.txt.gz ] && [ ! -e train.txt ];
then
    # download training dataset
    wget https://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
    # decompress the file
    gzip -d train.txt.gz
fi


if [ ! -e wiki-news-300d-1M.vec.zip ] && [ ! -e wiki-news-300d-1M.vec ];
then
    # download word embeddings
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
    # decompress the file
    unzip wiki-news-300d-1M.vec.zip
fi

