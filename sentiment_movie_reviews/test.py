from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import os
import lasagne
import cPickle
import sys
sys.path.append("../")
import networks
import argparse
import codecs
import data_iterator
import string

np.random.seed(2345)

#  SETTINGS
parser = argparse.ArgumentParser(description='Evaluate Sentiment Dataset.')
parser.add_argument("--settings_name", type=str,required=True,help="settings_file")
parser.add_argument("--model_folder", type=str,default="../models/sentiment_movie_reviews/",help="settings_file")
test_paras = vars(parser.parse_args())

file = codecs.open(os.path.join(test_paras["model_folder"],test_paras["settings_name"]),"r")
data = file.readlines()
paras = eval(data[0])


filename_model = string.replace(test_paras["settings_name"],"settings","data") + "_best"

matrices = cPickle.load(open(os.path.join(test_paras["model_folder"],filename_model),"rb"))


print(paras)

train_it, valid_it, test_it, matrix = data_iterator.load_model(paras)

print("-" * 80)
# Theano symbolic vars
sym_x = T.imatrix()
sym_mask = T.imatrix()
sym_y = T.ivector()


l_out = networks.densely_connected(sym_x, sym_mask, paras, matrix)

lasagne.layers.set_all_param_values(l_out,matrices)
#
valid_x_out = lasagne.layers.get_output(l_out, deterministic=True)
acc_valid = networks.accuracy_sigmoid(valid_x_out,sym_y,sym_mask,paras)


print("Compiling f_eval...")
fun_inp = [sym_x,sym_mask,sym_y]

fun_out = [acc_valid]

f_eval = theano.function(fun_inp,fun_out)


print("Started evaluating")
#########
# valid #
#########

valid_acc = 0
nr_of_batches = 0
mask_sum = 0
for x_batch, y_batch, mask_batch in test_it.get_batch(train=False):

    out = f_eval(x_batch,mask_batch,y_batch)

    valid_acc += np.sum(out[0])

    mask_sum += np.sum(mask_batch[:,0])

acc_valid = valid_acc / float(mask_sum)

print("Total Test Acc = " + str(acc_valid))



