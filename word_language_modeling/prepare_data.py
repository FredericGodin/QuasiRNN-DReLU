import os
import gzip
import codecs
import numpy as np

folder_ptb = '../data/word_language_modeling/ptb/'

def load_data(file_name, vocab_map, vocab_idx,gzipped=True,eow=True,prob=False):
    """
    Loads Penn Tree files downloaded from https://github.com/wojzaremba/lstm

    Notes
    -----
    This is python port of the LUA function load_data in
    https://github.com/wojzaremba/lstm/blob/master/data.lua

    Based on the orginal lasagne recipes implementation


    """
    def process_line(line):
        line = line.strip()
        words = line.split(" ")
        if words[-1] == "":
            del words[-1]
        if eow:
            words.append("<eos>")
        return words

    words = []
    if gzipped:
        with gzip.open(file_name, 'rb') as f:
            for line in f.readlines():
                words += process_line(line)
    else:
        with codecs.open(file_name, 'r',"UTF-8") as f:
            for line in f.readlines():
                words += process_line(line)

    n_words = len(words)
    print("Loaded %i words from %s" % (n_words, file_name))

    x = np.zeros(n_words)
    for wrd_idx, wrd in enumerate(words):
        if wrd not in vocab_map:
            vocab_map[wrd] = vocab_idx[0]
            vocab_idx[0] += 1
        x[wrd_idx] = vocab_map[wrd]



    counts = np.zeros((len(vocab_map),),dtype=np.float32)

    for wrd_idx, wrd in enumerate(words):
            counts[vocab_map[wrd]]+=1

    if prob:
        return x.astype('int32'), counts/len(words)
    else:
        return x.astype('int32'), None


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


def prepare_data(vocab_map, vocab_idx,filename,folder,prob=False):
    x = load_data(os.path.join(folder, filename),
                  vocab_map, vocab_idx,prob=prob)
    return x

def getdata(vocab_map = {},prob=False):
    vocab_idx = [0]
    train,train_prob = prepare_data(vocab_map, vocab_idx, "ptb.train.txt.gz",folder_ptb,prob=prob)
    valid,_ = prepare_data(vocab_map, vocab_idx, "ptb.valid.txt.gz",folder_ptb)
    test,_ = prepare_data(vocab_map, vocab_idx, "ptb.test.txt.gz",folder_ptb)

    if prob:
        return train,valid,test,vocab_map,vocab_idx,train_prob
    else:
        return train,valid,test,vocab_map,vocab_idx