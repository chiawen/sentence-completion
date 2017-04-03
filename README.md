# sentence-completion
These are Python3/Tensorflow implementations for MSR Sentence Completion Challenge.<br/>
<br/>
Code of RNN language model borrows heaveily from [word-rnn-tensorflow](https://github.com/hunkim/word-rnn-tensorflow).
## Requirements
- Python3
- Numpy
- Tensorflow 1.0
- NLTK 3.2.1
- SciPy
- scikit-learn
- gensim

## Data
Training and Test data set can be downloaded from the following link: <br/>
https://drive.google.com/open?id=0B5eGOMdyHn2mWDYtQzlQeGNKa2s <br/>
Google's pretrained word vectors can be downloaded here: [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) <br/>
<br/>
Please extract the training data and store them inside the `./data` directory.

## Overview of Models
### Recurrent Neural Netowrk Language Model (RNNLM)
- `utils.py`
- `model.py`
- `train.py`
- `inference.py`

### Total Word Similarity with Latent Semantic Analysis (LSA Total Simlilarity )
- `lsa_similariy.py`

### Total Word Similarity with [Google's pretrained word vectors](https://code.google.com/archive/p/word2vec/) (Word2vec Total Similarity)
- `word2vec_similarity.py`

I recommend to look at Platt's [Computational Approaches to Sentence Completion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/semco.pdf) paper.

## Basic Usage
To train the RNNLM model with default parameters, run:
```
python3 train.py
```
To generate a csv file of predictions from the latest saved checkpoint:
```
python3 inference.py
```

Train and output predictions using the LSA Total Similarity model:
```
python3 lsa_simlarity.py
```

Train and output predictions using the Word2vec Total Similarity model:
```
python3 word2vec_similarity.py
```

Calculate the average precision of predictions:
```
python3 acc.py -i [path_to_prediction_file]
```

## Pretrained Model
Generate predictions of the test set using pretrained RNN model:
```
bash ./run.sh
```
## Performance
|Method|Test|
|:---:|:---:|
|RNNLM|0.475|
|LSA Total Similarity|0.449|
|Word2vec Total Similarity|0.363|

