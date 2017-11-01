import os
import numpy
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import nltk
import codecs


def load_set(inverse_vocab_map, neg_pos, model):
    max_len = 0
    train_reviews = []

    word_count = len(inverse_vocab_map)

    for k in range(len(neg_pos)):
        path_train = os.path.join("../data/sentiment_movie_reviews/aclImdb", neg_pos[k])
        train_files = os.listdir(path_train)
        for file in train_files:
            review = codecs.open(os.path.join(path_train, file), "r", "UTF-8").readlines()[0].strip()
            words = nltk.word_tokenize(review)

            if len(words) > max_len:
                max_len = len(words)

            words_new = []
            for word in words:

                if len(word) == 0:
                    continue
                if word in model:
                    if word not in inverse_vocab_map:
                        inverse_vocab_map[word] = word_count
                        word_count += 1

                    words_new.append(inverse_vocab_map[word])

            train_reviews.append((words_new, k))

    x = numpy.zeros((len(train_reviews), max_len), dtype=numpy.int32)
    x_mask = numpy.zeros((len(train_reviews), max_len), dtype=numpy.int32)
    y = numpy.zeros((len(train_reviews),), dtype=numpy.int32)

    for i, (sentence, label) in enumerate(train_reviews):
        x[i, 0:len(sentence)] = sentence
        x_mask[i, 0:len(sentence)] = 1
        y[i] = label

    return x, x_mask, y


def load_data(model):
    inverse_vocab_map = {None: 0}

    neg_pos = ["train/neg/", "train/pos/"]
    x_train, x_mask_train, y_train = load_set(inverse_vocab_map, neg_pos, model)
    neg_pos = ["test/neg/", "test/pos/"]
    x_test, x_mask_test, y_test = load_set(inverse_vocab_map, neg_pos, model)

    return x_train, x_mask_train, y_train, x_test, x_mask_test, y_test, inverse_vocab_map


def make_embedding_matrix(inverse_vocab_map, model):
    matrix = np.zeros((len(inverse_vocab_map), model.vector_size), dtype=np.float32)

    for key, index in inverse_vocab_map.items():

        if key in model:
            matrix[index] = model[key]

    return matrix


if __name__ == "__main__":
    model = KeyedVectors.load_word2vec_format('../data/sentiment_movie_reviews/glove.840B.300d.w2v', binary=False)

    x_train, x_mask_train, y_train, x_test, x_mask_test, y_test, inverse_vocab_map = load_data(model)

    matrix = make_embedding_matrix(inverse_vocab_map, model)

    name = "glove.840B.300d.nltk_nounk"
    folder = "../prepared_data/sentiment_movie_reviews/"

    np.save(os.path.join(folder, name + ".x_train"), x_train)
    np.save(os.path.join(folder, name + ".x_mask_train"), x_mask_train)
    np.save(os.path.join(folder, name + ".y_train"), y_train)
    np.save(os.path.join(folder, name + ".x_test"), x_test)
    np.save(os.path.join(folder, name + ".x_mask_test"), x_mask_test)
    np.save(os.path.join(folder, name + ".y_test"), y_test)
    np.save(os.path.join(folder, name + ".matrix"), matrix)
