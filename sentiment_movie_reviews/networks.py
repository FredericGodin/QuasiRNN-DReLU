import lasagne
import theano.tensor as T

import layers
import initializers

def densely_connected(sym_x,sym_mask,paras,input_matrix):


    l_inp = lasagne.layers.InputLayer((paras["batch_size"], paras["model_seq_len"]),
                                      input_var=sym_x)

    W_input = input_matrix if input_matrix is not None else eval(paras["init_W"])


    l_emb = lasagne.layers.EmbeddingLayer(
        l_inp,
        input_size=paras["input_vocab_size"],  # size of embedding = number of words
        output_size=paras["embedding_size"],  # vector size used to represent each word
        W=W_input)

    l_emb = lasagne.layers.DropoutLayer(l_emb, p=paras["dropout_frac"])

    def create_gate(nonlinearity=eval(paras["gate_act"]), W_in=eval(paras["init_W"]),
                    W_cell=eval(paras["init_b"]), W_hid=eval(paras["init_W"]), b=eval(paras["init_b"])):
        return lasagne.layers.Gate(W_in=W_in, W_cell=W_cell, W_hid=W_hid, nonlinearity=nonlinearity, b=b)


    for i in range(paras["number_of_rnn_layers"]):

        if paras["rnn_type"] == "lstm":
            l_rec1 = lasagne.layers.LSTMLayer(
                l_emb,
                num_units=paras["rec_num_units"],
                ingate=create_gate(),
                forgetgate=create_gate(),
                cell=create_gate(W_cell=None, nonlinearity=eval(paras["input_act"])),
                outgate=create_gate(),
                nonlinearity=eval(paras["input_act"]),
                learn_init=(paras["learn_init"] > 0),
                backwards=False,
                only_return_final=( i == (paras["number_of_rnn_layers"]-1)),
                grad_clipping=paras["grad_clip"],
                peepholes=(paras["peepholes"] > 0),
                mask_input=lasagne.layers.InputLayer((paras["batch_size"], paras["model_seq_len"]),
                                                     input_var=sym_mask),
            )
        elif paras["rnn_type"] == "qrnn" or paras["rnn_type"] == "drelu" or paras["rnn_type"] == "delu":
            l_rec1 = layers.QRNNBlockSimplified(l_emb, paras, i,
                                         mask=lasagne.layers.InputLayer((paras["batch_size"], paras["model_seq_len"]),
                                                     input_var=sym_mask))

            if i == (paras["number_of_rnn_layers"]-1):
                l_rec1 = lasagne.layers.SliceLayer(l_rec1,-1,axis=1)

        l_rec1 = lasagne.layers.DropoutLayer(l_rec1, p=paras["dropout_frac"])

        if paras["dense"]:
            if i < paras["number_of_rnn_layers"] - 1:
                l_emb = lasagne.layers.ConcatLayer([l_emb,l_rec1],axis=2)
        else:
            l_emb = l_rec1



    x_out = lasagne.layers.DenseLayer(l_rec1,
                                      num_units=1,
                                      W=eval(paras["init_W"]),
                                      b=eval(paras["init_b"]),
                                      nonlinearity=lasagne.nonlinearities.sigmoid)



    return x_out

def accuracy_sigmoid(net_output, targets, mask, paras):
    net_output = T.flatten(net_output)
    targets = T.flatten(targets)
    correct = T.eq(net_output > 0.5,targets > 0.5)


    if mask is not None:
        return correct * mask[:,0]
    else:
        return correct

def binary_crossentropy(net_output, targets, mask = None):

    bce_loss = lasagne.objectives.binary_crossentropy(net_output.flatten(),targets.flatten())

    if mask is not None:
        return bce_loss.flatten() * mask[:,0]
    else:
        return bce_loss.flatten()

