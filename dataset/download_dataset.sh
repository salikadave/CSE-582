if [ ! -e train.txt.gz ] && [ ! -e train.txt ];
then
    # download training dataset
    wget https://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
    # decompress the file
    gzip -d train.txt.gz
fi