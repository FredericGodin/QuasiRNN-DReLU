import numpy as np
import theano
import theano.tensor as T
import os
import time
import lasagne
import cPickle
import sys
sys.path.append("../")

import networks
import argparse
import sys
import codecs

import data_iterator
from lasagne.regularization import l2
from lasagne.regularization import regularize_layer_params

np.random.seed(2345)

#  SETTINGS
parser = argparse.ArgumentParser(description='Sentiment Movie Review GPU version.')
parser.add_argument("--init_W", type=str, default="initializers.GlorotNormal()", help="initial parameter values")
parser.add_argument("--init_b", type=str, default="lasagne.init.Constant(0.0)", help="initial parameter values")
parser.add_argument("--input_act", type=str, default="lasagne.nonlinearities.tanh", help="activations of RNN")
parser.add_argument("--gate_act", type=str, default="lasagne.nonlinearities.sigmoid", help="gate activations of RNN")
parser.add_argument("--rec_num_units", type=int, default=256, help="number of LSTM units")
parser.add_argument("--dropout_frac", type=float, default=0.3, help="optional recurrent dropout")
parser.add_argument("--L2_reg", type=float, default=0.000004, help="L2 regularization")
parser.add_argument("--peepholes", type=int, default=0, help="Peephole connections in LSTM")
parser.add_argument("--untie_biases", type=int, default=0, help="Biases of QRNN")
parser.add_argument("--number_of_rnn_layers", type=int, default=4, help="Number of rnn layers")
parser.add_argument("--dense", type=int, default=1, help="densely connected or not")
parser.add_argument("--rnn_type", type=str, default="delu", help="lstm, qrnn, drelu, delu")
parser.add_argument("--k", type=int, default=[2, 2, 2, 2], nargs="+", help="k value qrnn")
parser.add_argument("--batch_norm", type=int, default=0, help="batchnorm")
parser.add_argument("--pooling", type=str, default="fo", help="f or fo")
parser.add_argument("--learn_init", type=int, default=0, help="learn init")
parser.add_argument("--elu_alpha", type=float, default=0.1, help="alpha value for elu")

# training
parser.add_argument("--batch_size", type=int, default=24, help=" batch size")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--optimizer", type=str, default="lasagne.updates.rmsprop", help="Optimizer function: sgd, adam...")
parser.add_argument("--max_grad_norm", type=float, default=5, help="scale steps if norm is above this value")
parser.add_argument("--grad_clip", type=float, default=0, help="Grad clipping value")
parser.add_argument("--num_epochs", type=int, default=60, help="Number of epochs to run")
parser.add_argument("--tol", type=float, default=1e-6, help="numerical stability")
parser.add_argument("--heldout", type=float, default=0.2, help="how much training is validation")
parser.add_argument("--heldout_index", type=int, default=5, help="Which subset will we use as validation")

# data
parser.add_argument("--load_file", type=str, default="glove.840B.300d.nltk_nounk")
parser.add_argument("--data_dir", type=str, default="../prepared_data/sentiment_movie_reviews/")
parser.add_argument("--save_dir", type=str, default="../models/sentiment_movie_reviews/")
parser.add_argument("--save_file", type=str, default="sentiment_")
args = parser.parse_args()
paras = vars(args)
print(paras)

paras["save_file"] += paras["load_file"]

train_it, valid_it, test_it, matrix = data_iterator.load_model(paras)

print("-" * 80)
# Theano symbolic vars
sym_x = T.imatrix()
sym_mask = T.imatrix()
sym_y = T.ivector()

sh_lr = theano.shared(lasagne.utils.floatX(paras["lr"]))

l_out = networks.densely_connected(sym_x, sym_mask, paras, matrix)

nr_of_params = lasagne.layers.count_params(l_out)
print("Nr of params: " + str(nr_of_params - matrix.shape[0] * matrix.shape[1]))
print("-" * 80)

# train
train_x_out = lasagne.layers.get_output(l_out, deterministic=False)
loss_train = T.mean(lasagne.objectives.binary_crossentropy(train_x_out, sym_y))
if paras["L2_reg"] > 0:
    loss_train += paras["L2_reg"] * regularize_layer_params(l_out, l2)

# valid
valid_x_out = lasagne.layers.get_output(l_out, deterministic=True)
loss_valid = networks.binary_crossentropy(valid_x_out, sym_y, sym_mask)
acc_valid = networks.accuracy_sigmoid(valid_x_out, sym_y, sym_mask, paras)

# remove embedding layer
all_params = lasagne.layers.get_all_params(l_out, trainable=True)
all_params = all_params[1:]

# gradients + updates
all_grads = T.grad(loss_train, all_params)
if paras["grad_clip"] > 0:
    all_grads_new = [T.clip(g, -paras["grad_clip"], paras["grad_clip"]) for g in all_grads]
    all_grads = all_grads_new
updates_gradients = eval(paras["optimizer"])(all_grads, all_params, learning_rate=sh_lr)

print("compiling f_eval...")
fun_inp = [sym_x, sym_mask, sym_y]

fun_out = [acc_valid, loss_valid]
f_eval = theano.function(fun_inp, fun_out)

print("compiling f_train...")
fun_out = [loss_train]
f_train = theano.function(fun_inp, fun_out, updates=updates_gradients)

print("Store settings")
start_time_str = time.strftime("%d_%b_%Y_%H_%M_%S")
save_file_model = paras["save_file"] + "_data_" + start_time_str
save_file_settings = paras["save_file"] + "_settings_" + start_time_str
file = codecs.open(os.path.join(paras["save_dir"], save_file_settings), "w")
file.write(str(paras) + "\n")
file.close()

print("started training")

best_valid_acc = 0
best_valid_loss = 1
for epoch in range(paras["num_epochs"]):

    #########
    # train #
    #########
    loss = 0
    nr_of_batches = 0
    batch_time = time.time()
    for x_batch, y_batch, mask_batch in train_it.get_batch():

        loss += f_train(x_batch, mask_batch, y_batch)[0]
        nr_of_batches += 1

        if nr_of_batches % 100 == 0:
            print("train: " + str(nr_of_batches))

    elapsed = time.time() - batch_time

    train_loss = loss / nr_of_batches

    print("Epoch " + str(epoch))
    print("Elapsed = " + str(int(elapsed)) + " seconds")
    print("Total Train loss = " + str(train_loss))

    #########
    # valid #
    #########

    valid_acc = 0
    valid_loss = 0
    nr_of_batches = 0
    mask_sum = 0
    for x_batch, y_batch, mask_batch in valid_it.get_batch(train=False):
        out = f_eval(x_batch, mask_batch, y_batch)

        valid_acc += np.sum(out[0])
        valid_loss += np.sum(out[1])
        mask_sum += np.sum(mask_batch[:, 0])

    elapsed = time.time() - batch_time

    loss_valid = valid_loss / float(mask_sum)
    print("Total Valid Loss = " + str(loss_valid))
    acc_valid = valid_acc / float(mask_sum)
    print("Total Valid Acc = " + str(acc_valid))

    if acc_valid > best_valid_acc:
        print("----New best accuracy: " + str(acc_valid))
        best_valid_acc = acc_valid

        with open(os.path.join(paras["save_dir"], save_file_model + "_best"), 'w') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(l_out), f,
                         cPickle.HIGHEST_PROTOCOL)

    with codecs.open(os.path.join(paras["save_dir"], save_file_settings), "a") as f:
        f.write("Epoch %s, valid perplexity %s, train_perplexity %s\n" % (
            epoch + 1, acc_valid, train_loss))

with open(os.path.join(paras["save_dir"], save_file_model + "_last"), 'wb') as f:
    cPickle.dump(lasagne.layers.get_all_param_values(l_out), f,
                 cPickle.HIGHEST_PROTOCOL)
