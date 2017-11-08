# Quasi-Recurrent Neural Networks with Dual Rectified Linear Units
This repository contains an implementation of Quasi-Recurrent Neural Networks which matches the results mentioned in the original paper.
Additionally, we introduce a novel unit, called DReLUs which can replace the tanh activation function in QRNNs.
DReLUs are less prone to vanishing gradients and consequently effectuate efficient training of QRNNs.
To the best of our knowledge, this is one of the first state-of-the-art attempts to introduce ReLU-based activation functions in the recurrent step of a RNN variant.
A full explanation and discussion can be found in our paper:
["Dual Rectified Linear Units (DReLUs): A Replacement for Tanh Activation Functions in Quasi-Recurrent Neural Networks"](https://arxiv.org/pdf/1707.08214v2.pdf)

## Installation

The following software was used:

  * Python 2.7.12
  * Theano 0.9.0b1
  * Lasagne 0.2.dev1
  * Gensim 2.1.0
  * Nltk 3.2.4

## Sentiment Classification
To reproduce the sentiment classification experiments, you need to download the following files:

Download the [Imdb Movie Review dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and unzip it in the folder data/sentiment_movie_reviews/

```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz
```
Download the [GloVe embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and unzip them in the folder data/sentiment_movie_reviews/.
Next convert them to a Gensim readable format.

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
python -m gensim.scripts.glove2word2vec --input glove.840B.300d.txt --output glove.840B.300d.w2v
```


First, you need to prepare the data for training the network.
In practice, we extract all Glove embeddings of words which are present in the movie review dataset.
Also, we turn the movie reviews in matrices of word indexes.
```
python prepare_data.py
```

To train the models, there are 3 parameters you can change to reproduce the results of the paper:

  * --rnn_type lstm, qrnn, drelu or delu
  * --rec_num_units any_number
  * --dense 0,1

For example, for the orginal QRNN:
```
THEANO_FLAGS=device=gpu0,floatX=float32 python train.py --rnn_type qrnn --dense 1 --rec_num_units 242
```

Finally, for obtaining the test result we need to select the model we have trained and provide it to the script:
```
THEANO_FLAGS=device=gpu0,floatX=float32 python test.py --settings_name sentiment_glove.840B.300d.nltk_nounk_settings_DATE
```

## Word-level language modeling
The word-level language modeling experiments are executed on the Penn Treebank dataset.
Typically, the standard splits of Mikolov et al. 2012 are used.
The files are already uploaded in data/word_language_modeling/ptb/ folder.

For reproducing the result of Zaremba et al. 2014.
```
THEANO_FLAGS=device=gpu0,floatX=float32 python train.py --rnn_type lstm --rec_num_units 650 --embedding_size 650 --decay 0.83333 --model_seq_len 35 --max_grad_norm 5
```






