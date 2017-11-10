import lasagne
import layers
import numpy as np
import theano.tensor as T


def build_network(paras,sym_x,hids):

    if paras["rnn_type"]=="lstm":
        l_inp = lasagne.layers.InputLayer((paras["batch_size"], paras["model_seq_len"]),
                                          input_var=sym_x)
    else:
        l_inp = lasagne.layers.InputLayer((paras["batch_size"], paras["model_seq_len"]+paras["k"][0]-1),input_var=sym_x)

    l_emb = lasagne.layers.EmbeddingLayer(
        l_inp,
        input_size=paras["vocab_size"],       # size of embedding = number of words
        output_size=paras["embedding_size"],  # vector size used to represent each word
        W=eval(paras["init_W"]))


    def create_gate(nonlinearity=eval(paras["gate_act"]), W_in=eval(paras["init_W"]),
                    W_cell=eval(paras["init_W"]), W_hid=eval(paras["init_W"]),b=eval(paras["init_b"])):
        return lasagne.layers.Gate(W_in=W_in, W_cell=W_cell, W_hid=W_hid, nonlinearity=nonlinearity,b=b)


    nr_of_hids = 2
    hid_i=0
    hids_out = []

    for i in range(paras["number_of_rnn_layers"]):


        if paras["rnn_type"] == "lstm":

            rec_out = layers.lstm_layers.LSTMLayer(
                l_emb,
                num_units=paras["rec_num_units"],
                ingate=create_gate(),
                forgetgate=create_gate(),
                cell=create_gate(W_cell=None, nonlinearity=eval(paras["input_act"])),
                outgate=create_gate(),
                nonlinearity=eval(paras["input_act"]),
                learn_init=False,
                backwards=False,
                grad_clipping=paras["grad_clip"],
                peepholes=(paras["peepholes"] > 0),
                hid_init=lasagne.layers.InputLayer((paras["batch_size"],paras["rec_num_units"]),input_var=hids[2*i]),
                cell_init=lasagne.layers.InputLayer((paras["batch_size"],paras["rec_num_units"]),input_var=hids[2*i+1])

            )

            l_final_cell_1 = layers.custom_layers.SelectOutputLayer(rec_out, 1)
            l_rec1 = layers.custom_layers.SelectOutputLayer(rec_out, 0)

            hids_out.append(lasagne.layers.SliceLayer(l_rec1, indices=-1, axis=1))
            hids_out.append(lasagne.layers.SliceLayer(l_final_cell_1, indices=-1, axis=1))

            if i < paras["number_of_rnn_layers"] - 1:
                l_rec1 = lasagne.layers.DropoutLayer(l_rec1, p=paras["dropout_frac"])

            l_emb = l_rec1

        elif paras["rnn_type"] == "qrnn" or paras["rnn_type"] == "drelu" or paras["rnn_type"] == "delu":

            rec_out = layers.QRNNBlock(l_emb, paras, i, mask=None, hids=hids)

            l_rec1 = rec_out[0]
            l_rec1_cell = rec_out[1]


            if i<paras["number_of_rnn_layers"]-1:
                l_rec1_hids_to_forward = lasagne.layers.SliceLayer(l_rec1, slice(paras["model_seq_len"] - paras["k"][i+1] + 1,
                                                                               paras["model_seq_len"]), axis=1)
                hids_out.append(l_rec1_hids_to_forward)

            l_final_cell_out_1 = lasagne.layers.SliceLayer(l_rec1_cell, indices=-1, axis=1)
            hids_out.append(l_final_cell_out_1)

            if i < paras["number_of_rnn_layers"] - 1:
                previous_hids = lasagne.layers.InputLayer((paras["batch_size"],paras["k"][i+1]-1,paras["rec_num_units"]),input_var=hids[nr_of_hids*hid_i+0])

                # will only be applied in between layers
                previous_hids = lasagne.layers.DropoutLayer(previous_hids, p=paras["dropout_frac"])
                l_rec1 = lasagne.layers.DropoutLayer(l_rec1, p=paras["dropout_frac"])

                l_emb = lasagne.layers.ConcatLayer([previous_hids, l_rec1], axis=1)

                hid_i+=1

    l_rec1 = lasagne.layers.DropoutLayer(l_rec1, p=paras["dropout_frac"])

    l_shp = lasagne.layers.ReshapeLayer(l_rec1,
                                        (paras["batch_size"]*paras["model_seq_len"], paras["rec_num_units"] ))
    l_out = lasagne.layers.DenseLayer(l_shp,
                                      num_units=paras["vocab_size"],
                                      W=eval(paras["init_W"]),
                                      b=eval(paras["init_b"]),
                                      nonlinearity=None)

    l_out = lasagne.layers.NonlinearityLayer(l_out,nonlinearity=lasagne.nonlinearities.softmax)

    l_out = lasagne.layers.ReshapeLayer(l_out,
                                        (paras["batch_size"], paras["model_seq_len"], paras["vocab_size"]))



    return l_out, hids_out


def calc_bpc_variable(x,y,paras,f_eval,padding=None):

    n_batches = x.shape[0] / paras["batch_size"]
    l_cost = []

    hids = []
    if paras["rnn_type"]=="lstm":
        for i in range(paras["number_of_rnn_layers"]):
            hids.append(np.zeros((paras["batch_size"], paras["rec_num_units"]),
                                 dtype=np.float32))
            hids.append(np.zeros((paras["batch_size"], paras["rec_num_units"]),
                                 dtype=np.float32))
    else:
        for i in range(paras["number_of_rnn_layers"]-1):

            hids.extend([np.zeros((paras["batch_size"],paras["k"][(i+1) % len(paras["k"])]-1, paras["rec_num_units"]),
                               dtype=np.float32), np.zeros((paras["batch_size"], paras["rec_num_units"]),
                               dtype=np.float32)])


        hids.append( np.zeros((paras["batch_size"], paras["rec_num_units"]),
                               dtype=np.float32))
    function_str = "f_eval("
    for j in range(2+len(hids)):
        function_str+="input["+str(j)+"],"
    function_str=function_str[:-1]+")"

    for i in range(n_batches):
        x_batch = x[i*paras["batch_size"]:(i+1)*paras["batch_size"]]
        y_batch = y[i*paras["batch_size"]:(i+1)*paras["batch_size"]]

        mask_batch = padding[i*paras["batch_size"]:(i+1)*paras["batch_size"]]

        input = [x_batch, y_batch]
        input.extend(hids)
        all = eval(function_str)
        cost = all[0]
        hids = all[1:]

        if padding is not None:
            cost = np.sum(cost*mask_batch.flatten())
        else:
            cost = np.mean(cost)


        l_cost.append(cost)


    if padding is not None:
        perplexity = (np.sum(l_cost) / (np.sum(padding)))
    else:
        perplexity =(np.sum(l_cost) / len(l_cost))

    return perplexity

def calc_cross_ent_bpc(net_output, targets,paras):
    # Helper function to calculate the cross entropy error
    preds = T.reshape(net_output, (paras["batch_size"] * paras["model_seq_len"], -1))
    preds += paras["tol"]  # add constant for numerical stability
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)/T.log(2)
    return cost