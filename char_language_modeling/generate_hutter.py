import codecs
import numpy as np
import cPickle

def generate_hutter(path = "../data/language_modeling/hutter/enwik8",vocab_map = {}):

    # based on https://github.com/julian121266/RecurrentHighwayNetworks/blob/master/data/create_hutter_enwik8.py
    num_test_chars = 5000000

    print("Extracting Hutter Prize data ...")
    with codecs.open(path, 'r') as f:
        raw_data = f.read()
    print("Done.")

    print("Preparing data for Brainstorm ...")
    raw_data = np.fromstring(raw_data, dtype=np.uint8)
    unique, data = np.unique(raw_data, return_inverse=True)

    print("Vocabulary size:", unique.shape)

    x = np.zeros(data.shape)
    vocab_idx=0
    for wrd_idx, wrd in enumerate(data):
        if wrd not in vocab_map:
            vocab_map[wrd] = vocab_idx
            vocab_idx += 1

        x[wrd_idx] = vocab_map[wrd]

        if wrd_idx % 10000 == 0:
            print(wrd_idx)


    train_data = x[: -2 * num_test_chars]
    valid_data = x[-2 * num_test_chars: -num_test_chars]
    test_data = x[-num_test_chars:]

    return train_data.astype('int32'),valid_data.astype('int32'),test_data.astype('int32'),vocab_map



if __name__ == "__main__":
    train,valid,test,vocab_map = generate_hutter()

    np.save(open("../prepared_data/char_language_modeling/hutter_train.npy","wb"),train)
    np.save(open("../prepared_data/char_language_modeling/hutter_valid.npy", "wb"), valid)
    np.save(open("../prepared_data/char_language_modeling/hutter_test.npy", "wb"), test)
    cPickle.dump(vocab_map,open("../prepared_data/char_language_modeling/hutter_vocab.pkl","wb"))