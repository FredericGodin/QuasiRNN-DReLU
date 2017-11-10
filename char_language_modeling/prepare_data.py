
import codecs
import gzip
import os
import numpy as np
import cPickle


folder_hutter = '../data/hutter/'

def getdata(folder_ptb = "../data/language_modeling/ptb/",vocab_map = {}):

    train = load_data(os.path.join(folder_ptb,"ptb.train.txt.gz"),vocab_map, train=True)
    valid = load_data(os.path.join(folder_ptb,"ptb.valid.txt.gz"),vocab_map, train=False)
    test= load_data(os.path.join(folder_ptb,"ptb.test.txt.gz"),vocab_map, train=False)

    return train,valid,test,vocab_map


def getdata_hutter(folder_hutter = "../data/language_modeling/hutter/"):

    with open(os.path.join(folder_hutter,"hutter_train.npy"),"rb") as f:
        train = np.load(f)
    with open(os.path.join(folder_hutter,"hutter_valid.npy"),"rb") as f:
        valid = np.load(f)
    with open(os.path.join(folder_hutter,"hutter_test.npy"),"rb") as f:
        test = np.load(f)

    return train,valid,test

def load_data(file_name, vocab_map, gzipped=True,train=True):

    def process_line(line):

        return list(line[:-2])

    words = []
    len_total = 0
    if gzipped:
        with gzip.open(file_name, 'rb') as f:
            for line in f.readlines():
                new_words = process_line(line)
                len_total += len(new_words)
                words += new_words
                #
                # if len_total > 64000:
                #     break
    else:
        with codecs.open(file_name, 'r',"UTF-8") as f:
            for line in f.readlines():
                words += process_line(line)

    n_words = len(words)
    print("Loaded %i chars from %s" % (n_words, file_name))

    x = np.zeros(n_words)
    vocab_idx=0
    for wrd_idx, wrd in enumerate(words):
        if wrd not in vocab_map:
            if train:
                vocab_map[wrd] = vocab_idx
                vocab_idx += 1
            else:
                print("new char?")
        x[wrd_idx] = vocab_map[wrd]


    return x.astype('int32')

def reorder(x_in, batch_size, model_seq_len,padding=False,context=0):
    """
    Rearranges data set so batches process sequential data.
    If wed have the dataset:

    x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

    and the batch size is 2 and the model_seq_len is 3. Then the dataset is
    reordered such that:

                   Batch 1    Batch 2
                 ------------------------
    batch pos 1  [1, 2, 3]   [4, 5, 6]
    batch pos 2  [7, 8, 9]   [10, 11, 12]

    This ensures that we use the last hidden state of batch 1 to initialize
    batch 2.

    Also creates targets. In language modelling the target is to predict the
    next word in the sequence.

    Parameters
    ----------
    x_in : 1D numpy.array
    batch_size : int
    model_seq_len : int
        number of steps the model is unrolled

    Returns
    -------
    reordered x_in and reordered targets. Targets are shifted version of x_in.

    """
    if x_in.ndim != 1:
        raise ValueError("Data must be 1D, was", x_in.ndim)

    in_len = len(x_in) - 1


    x_resize =  \
        (in_len / (batch_size*model_seq_len))*model_seq_len*batch_size

    if padding and x_resize != in_len:
        x_resize+=batch_size*model_seq_len

    n_samples = x_resize / (model_seq_len)
    n_batches = n_samples / batch_size

    targets=np.empty(x_resize)
    targets.fill(-1)
    x_out = np.zeros(x_resize)

    copy_len = min(in_len,x_resize)
    targets[0:copy_len] = x_in[1:copy_len+1]
    targets = targets.reshape(n_samples, model_seq_len)
    x_out[0:copy_len] = x_in[:copy_len]
    x_out = x_out.reshape(n_samples, model_seq_len)

    if padding:
        mask = np.zeros(x_resize)
        mask[0:copy_len]=1
        mask = mask.reshape(n_samples, model_seq_len)



    out = np.zeros(n_samples, dtype=int)
    for i in range(n_batches):
        val = range(i, n_batches*batch_size+i, n_batches)
        out[i*batch_size:(i+1)*batch_size] = val

    x_out = x_out[out]
    targets = targets[out]


    if context > 0:
        # prolong model_seq_len
        x_long = np.zeros((x_out.shape[0],x_out.shape[1]+context))
        x_long[:,context:]=x_out

        len_seq = x_out.shape[1]
        for i in range(x_out.shape[0]-batch_size):
            x_long[batch_size+i,0:context]=x_out[i,(len_seq-context):]

        x_out = x_long


    if padding:
        mask = mask[out]
        return x_out.astype('int32'), targets.astype('int32'),mask.astype('float32')
    else:
        return x_out.astype('int32'), targets.astype('int32')




