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
import string
import prepare_data

np.random.seed(2345)

#  SETTINGS
parser = argparse.ArgumentParser(description='Evaluate Sentiment Dataset.')
parser.add_argument("--settings_name", type=str,required=True,help="Settings file of the format lm_char_settings_DATE")
parser.add_argument("--model_folder", type=str,default="../models/char_language_modeling/",help="Model folder containing the settings file")
test_paras = vars(parser.parse_args())

file = codecs.open(os.path.join(test_paras["model_folder"],test_paras["settings_name"]),"r")
data = file.readlines()
paras = eval(data[0])


filename_model = string.replace(test_paras["settings_name"],"settings","data") + "_best"

matrices = cPickle.load(open(os.path.join(test_paras["model_folder"],filename_model),"rb"))


sym_x = T.imatrix()
sym_y = T.imatrix()

hids = []

for _ in range(paras["number_of_rnn_layers"]):
    if paras["rnn_type"] == "lstm":
        hids.extend([T.matrix(), T.matrix()])
    else:
        hids.extend([T.tensor3(), T.matrix()])

l_out, l_hids = networks.build_network(paras, sym_x, hids)

lasagne.layers.set_all_param_values(l_out,matrices)

all_out = [l_out]
all_out.extend(l_hids)

eval_out = lasagne.layers.get_output(
    all_out, deterministic=True)
hids_out_eval = eval_out[1:]
eval_out = eval_out[0]

cost_eval = networks.calc_cross_ent_bpc(eval_out, sym_y, paras)


print("Compiling f_eval...")
fun_inp = [sym_x, sym_y]

if paras["rnn_type"] != "lstm":
    hids.pop(-2)

fun_inp.extend(hids)
outs = [cost_eval]
outs.extend(hids_out_eval)

f_eval = theano.function(fun_inp, outs)

print("Loading data...")
if paras["dataset"]=="hutter":
    train, valid, test = prepare_data.getdata_hutter()
    paras["vocab_size"] = 205
else:
    train, valid, test, vocab_map = prepare_data.getdata()
    paras["vocab_size"] = 49

x_test,y_test,mask_test = prepare_data.reorder(test,paras["batch_size"],paras["model_seq_len"],padding=True,context=paras["k"][0]-1)

print("Started testing")
bpc_test = networks.calc_bpc_variable(x_test, y_test, paras, f_eval, padding=mask_test)

print("Test BPC: %s " % (bpc_test))
