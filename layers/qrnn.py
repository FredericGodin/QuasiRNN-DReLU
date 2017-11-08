__author__ = 'frederic'

import theano
import theano.tensor as T
import lasagne
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer

import initializers
import custom_layers

def neg_rectify(x):
    return -1.0*T.nnet.relu(x)

def rectify(x):
    return T.nnet.relu(x)


class ELU(object):
    def __init__(self, alpha=1.0,neg=False):
        self.alpha = alpha
        self.neg = neg

    def __call__(self, x):

        direction = -1.0 if self.neg else 1.0

        return direction*theano.tensor.switch(x > 0, x, self.alpha * theano.tensor.expm1(x))

class QRNNLayer(MergeLayer):


    def __init__(self, incoming,gate_pos, num_units,
                 seq_len=0,
                 cell_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 only_return_final=False,
                 mask_input=None,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming,gate_pos]
        self.cell_init_incoming_index = -1
        self.mask_incoming_index = -1

        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(QRNNLayer, self).__init__(incomings, **kwargs)


        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.only_return_final=only_return_final
        self.seq_len=seq_len

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")


        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)


    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened

        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return (input_shape[0], input_shape[2], self.num_units),(input_shape[0], input_shape[2], self.num_units)

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        gate = inputs[1]

        cell_init = None
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input[:, :, :, 0].dimshuffle(2, 0, 1)
        gate = gate[:, :, :, 0].dimshuffle(2, 0, 1)



        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation


        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, gate_n,hid_previous,*args):

            hid=input_n
            # temp=rectify( gate_pos_n*rectify(hid_previous) )
            # temp+=neg_rectify_neg( gate_neg_n*neg_rectify_neg(hid_previous) )

            # temp = T.nnet.hard_sigmoid(gate_n)*hid_previous
            temp = gate_n*hid_previous

            hid += temp


            return hid,temp

        def step_masked(input_n, gate_n, mask_n, hid_previous, *args):
            hid, temp = step(input_n, gate_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid, temp

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, gate, mask]
            step_fun = step_masked
        else:
            sequences = [input, gate]
            step_fun = step


        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)


        outputs_info = [cell_init, None]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            outs = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                non_sequences=[],
                outputs_info=outputs_info,
                go_backwards=self.backwards,
                n_steps=self.seq_len)
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            outs = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=outputs_info,
                go_backwards=self.backwards,
                #truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.only_return_final:
            return outs[-1]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        cell_out = outs[0].dimshuffle(1, 0, 2)
        temp_out = outs[1].dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            cell_out = cell_out[:, ::-1]

        return cell_out,temp_out

def QRNNBlock(l_in,paras,i,mask,hids):


    if i == 0:
        input_size = paras["embedding_size"]
    else:
        if "dense" in paras and paras["dense"]:
            input_size = paras["embedding_size"] + i*paras["rec_num_units"]
        else:
            input_size = paras["rec_num_units"]

    if hids == None:
        # if no hids are passed, we are in the single sentence case and need to pad the input ourself
        l_in = lasagne.layers.PadLayer(l_in,((paras["k"][i]-1,0),(0,0)),batch_ndim=1)


    l_emb_reshaped = lasagne.layers.ReshapeLayer(l_in, (
        paras["batch_size"], 1, -1, input_size))

    l_conv_gates_rec = lasagne.layers.Conv2DLayer(l_emb_reshaped,  paras["rec_num_units"],
                                               (paras["k"][i], input_size), pad="valid", W=eval(paras["init_W"]),
                                               b=eval(paras["init_b"]), untie_biases=paras["untie_biases"],
                                               nonlinearity=nonlinearities.identity,name="forget_gate")

    if paras["batch_norm"]==1:
        l_conv_gates_rec = lasagne.layers.batch_norm(l_conv_gates_rec,gamma=lasagne.init.Constant(0.1))

    l_conv_gates_rec_hidden = lasagne.layers.NonlinearityLayer(l_conv_gates_rec,nonlinearity=eval(paras["gate_act"]))

    l_conv_gates_rec_input = lasagne.layers.NonlinearityLayer(l_conv_gates_rec_hidden,nonlinearity=(lambda x: 1.0-x))

    if paras["rnn_type"] == "qrnn":

        l_conv_input = lasagne.layers.Conv2DLayer(l_emb_reshaped, paras["rec_num_units"], (paras["k"][i], input_size),
                                                   pad="valid", W=eval(paras["init_W"]), b=eval(paras["init_b"]),
                                                   untie_biases=paras["untie_biases"], nonlinearity=eval(paras["input_act"]),name="input_1")

        if paras["batch_norm"] > 0:
            l_conv_input = lasagne.layers.batch_norm(l_conv_input,gamma=lasagne.init.Constant(0.1))

    elif paras["rnn_type"] == "drelu" or paras["rnn_type"] == "delu":

        if paras["rnn_type"] == "delu":
            act1 = ELU(paras["elu_alpha"])
            act2 = ELU(paras["elu_alpha"], neg=True)
        else:
            act1 = rectify
            act2 = neg_rectify

        l_conv_input1 = lasagne.layers.Conv2DLayer(l_emb_reshaped, paras["rec_num_units"], (paras["k"][i], input_size),
                                                   pad="valid", W=eval(paras["init_W"]), b=eval(paras["init_b"]),
                                                   untie_biases=paras["untie_biases"], nonlinearity=act1, name="input_1")
        l_conv_input2 = lasagne.layers.Conv2DLayer(l_emb_reshaped, paras["rec_num_units"], (paras["k"][i], input_size),
                                                   pad="valid", W=eval(paras["init_W"]), b=eval(paras["init_b"]),
                                                   untie_biases=paras["untie_biases"], nonlinearity=act2, name="input_2")

        if paras["batch_norm"] > 0:
            l_conv_input1 = lasagne.layers.batch_norm(l_conv_input1, gamma=lasagne.init.Constant(0.5))
            l_conv_input2 = lasagne.layers.batch_norm(l_conv_input2, gamma=lasagne.init.Constant(0.5))

        # Z
        l_conv_input = lasagne.layers.ElemwiseSumLayer([l_conv_input1, l_conv_input2], name="sum_input")


    l_conv_input_gated = lasagne.layers.ElemwiseMergeLayer([l_conv_input, l_conv_gates_rec_input], T.mul)


    if hids == None:
        cell_init = lasagne.init.Constant(0)
    else:
        cell_init = lasagne.layers.InputLayer((paras["batch_size"], paras["rec_num_units"]),
                                            input_var=hids[2 * i + 1])

    l_rec1_cells = QRNNLayer(
        l_conv_input_gated,l_conv_gates_rec_hidden,
        num_units=paras["rec_num_units"],
        learn_init=False,
        mask_input=mask,
        cell_init = cell_init
    )

    l_rec_1 = custom_layers.SelectOutputLayer(l_rec1_cells,0)

    if paras["pooling"]=="fo":

        init_f = eval(paras["init_W"])
        l_conv_gates_out = lasagne.layers.Conv2DLayer(l_emb_reshaped, paras["rec_num_units"],
                                                      (paras["k"][i], input_size), pad="valid", W=init_f,
                                                      b=eval(paras["init_b"]), untie_biases=paras["untie_biases"],
                                                      nonlinearity=nonlinearities.identity, name="out_gate")

        if paras["batch_norm"]==1:
            l_conv_gates_out = lasagne.layers.batch_norm(l_conv_gates_out,gamma=lasagne.init.Constant(0.1))


        l_conv_gates_out = lasagne.layers.NonlinearityLayer(l_conv_gates_out,nonlinearity=eval(paras["gate_act"]))

        l_conv_gates_out = lasagne.layers.SliceLayer(l_conv_gates_out,0,axis=3)

        l_conv_gates_out = lasagne.layers.DimshuffleLayer(l_conv_gates_out,(0,2,1),name="out_neg_gate")

        l_rec1_hids = lasagne.layers.ElemwiseMergeLayer([l_rec_1,l_conv_gates_out],T.mul)
    else:
        l_rec1_hids = l_rec_1


    return l_rec1_hids, l_rec_1


















