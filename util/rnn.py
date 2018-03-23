from __future__ import absolute_import, division, print_function

import tensorflow as tf

def dec_lstm_layer(name, inSeq_embeddings, sequence_length, vocab_size, image_embeddings, input_dim, keep_prob=1.0):
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).

    with tf.variable_scope(name, initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08)):
        lstm_cell = tf.contrib.rnn.LSTMCell( 
                         input_dim, forget_bias=1.0, state_is_tuple=True)
        if keep_prob < 1.0:
            lstm_cell = tf.contrib.rnn.DropoutWrapper( lstm_cell,
                            input_keep_prob=keep_prob, output_keep_prob=keep_prob)

        # Feed the image embeddings to set the initial LSTM state.
        zero_state = lstm_cell.zero_state( batch_size=image_embeddings.get_shape()[0],
                    dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)
        # Allow the LSTM variables to be reused.
    with tf.variable_scope(name, reuse=True) as scope:
        # Run the batch of sequence embeddings through the LSTM.
        lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=inSeq_embeddings,
            sequence_length=sequence_length, initial_state=initial_state, dtype=tf.float32, scope=scope)
        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    logits = fc("logits",lstm_outputs, vocab_size,
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

    return logits    

def lstm_dynamic_layer(name, seq_bottom, const_bottom, seq_len, output_dim,
                       num_layers=1,forget_bias=0.0, apply_dropout=False, keep_prob=0.5,
                       concat_output=True, initializer=None):
    batch_size = seq_bottom.get_shape().as_list()[0]
    with tf.variable_scope(name) as scope:
        lstm_cell = tf.contrib.rnn.LSTMCell(output_dim, forget_bias=forget_bias, initializer=initializer)
        # Apply dropout if specified.
        if apply_dropout:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

        # Initialize cell state from zero.
        initial_state = cell.zero_state(batch_size, tf.float32)

        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=seq_bottom,
            sequence_length=seq_len, initial_state=initial_state, dtype=tf.float32, scope=scope)    
        return outputs, states                     

def lstm_layer(name, seq_bottom, const_bottom, output_dim, num_layers=1,
               forget_bias=0.0, apply_dropout=False, keep_prob=0.5,
               concat_output=True, initializer=None):
    """
    Similar LSTM layer as the `LSTMLayer` in Caffe
    ----
    Args:
        seq_bottom : the underlying sequence input of size [T, N, D_in], where
            D_in is the input dimension, T is num_steps and N is batch_size.
        const_bottom : the constant bottom concatenated to each time step,
            having shape [N, D_const]. This can be *None*. If it is None,
            then this input is ignored.
        output_dim : the number of hidden units in the LSTM unit and also the
            final output dimension, i.e. D_out.
        num_layers : the number of stacked LSTM layers.
        forget_bias : forget gate bias in LSTM unit.
        apply_dropout, keep_prob: dropout applied to the output of each LSTM
            unit.
    Returns:
        output : a list of [T, N, D_out], where D_out is output_dim,
            T is num_steps and N is batch_size
    """

    # input shape is [T, N, D_in]
    input_shape = seq_bottom.get_shape().as_list()
    # the number of time steps to unroll
    num_steps = input_shape[0]
    # batch size (i.e. N)
    batch_size = input_shape[1]

    # The actual parameter variable names are as follows (`name` is the name
    # variable here, and Cell0, Cell1, ... are num_layers stacked LSTM cells):
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias
    # where Cell1 is on top of Cell0, taking Cell0's hidden states as inputs.
    #
    # For Cell0, the weight matrix ('BasicLSTMCell/Linear/Matrix') has shape
    # [D_in+D_const+D_out, 4*D_out], and bias has shape [4*D_out].
    # For Cell1, Cell2, ..., the weight matrix ('BasicLSTMCell/Linear/Matrix')
    # has shape [D_out*2, 4*D_out], and bias has shape [4*D_out].
    # In the weight matrix, the first D_in+D_const rows (in Cell0) or D_out rows
    # (in Cell1, Cell2, ...) are bottom input weights, and the rest D_out rows
    # are state weights, i.e. *inputs are before states in weight matrix*
    #
    # The gate order in 4*D_out are i, j, f, o, where
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    #
    # Other details in tensorflow/python/ops/rnn_cell.py
    with tf.variable_scope(name):
        # the basic LSTM cell
        lstm_cell = tf.contrib.rnn.LSTMCell(output_dim, forget_bias=forget_bias, initializer=initializer)
        # Apply dropout if specified.
        if apply_dropout:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

        # Initialize cell state from zero.
        initial_state = cell.zero_state(batch_size, tf.float32)
        # Fix batch_size issue when batch_size == 1
        # state_shape = initial_state.get_shape().as_list()
        # state_shape[0] = batch_size
        # initial_state.set_shape(state_shape)

        # Split along time dimension and flatten each component.
        # `inputs` is a list.
        inputs = [tf.reshape(input_, [batch_size, -1])
            for input_ in tf.split(seq_bottom, num_steps, 0)]
        # Add constant input to each time step.
        if not const_bottom is None:
            # Flatten const_bottom into shape [N, D_const] and concatenate.
            const_input_ = tf.reshape(const_bottom, [batch_size, -1])
            inputs = [tf.concat(0, [input_, const_input_])
                for input_ in inputs]

        # Create the Recurrent Network and collect `outputs` and `states`
        outputs, states = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=initial_state)
        if concat_output:
            # Concat the outputs into [T, N, D_out].
            outputs = tf.reshape(tf.concat(0, outputs),
                                [num_steps, batch_size, output_dim])
    return outputs, states

def bi_lstm_layer(name, seq_bottom, const_bottom, output_dim, num_layers=1,
               forget_bias=0.0, apply_dropout=False, keep_prob=0.5,
               concat_output=True, initializer=None):
    """
    Bi-directional LSTM layer
    ----
    Args:
        seq_bottom : the underlying sequence input of size [T, N, D_in], where
            D_in is the input dimension, T is num_steps and N is batch_size.
        const_bottom : the constant bottom concatenated to each time step,
            having shape [N, D_const]. This can be *None*. If it is None,
            then this input is ignored.
        output_dim : the number of hidden units in the LSTM unit and also the
            final output dimension, i.e. D_out.
        num_layers : the number of stacked LSTM layers.
        forget_bias : forget gate bias in LSTM unit.
        apply_dropout, keep_prob: dropout applied to the output of each LSTM
            unit.
    Returns:
        output : a list of [T, N, D_out], where D_out is output_dim,
            T is num_steps and N is batch_size
    """

    # input shape is [T, N, D_in]
    input_shape = seq_bottom.get_shape().as_list()
    # the number of time steps to unroll
    num_steps = input_shape[0]
    # batch size (i.e. N)
    batch_size = input_shape[1]

    # The actual parameter variable names are as follows (`name` is the name
    # variable here, and Cell0, Cell1, ... are num_layers stacked LSTM cells):
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix
    #   `name`/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias
    # where Cell1 is on top of Cell0, taking Cell0's hidden states as inputs.
    #
    # For Cell0, the weight matrix ('BasicLSTMCell/Linear/Matrix') has shape
    # [D_in+D_const+D_out, 4*D_out], and bias has shape [4*D_out].
    # For Cell1, Cell2, ..., the weight matrix ('BasicLSTMCell/Linear/Matrix')
    # has shape [D_out*2, 4*D_out], and bias has shape [4*D_out].
    # In the weight matrix, the first D_in+D_const rows (in Cell0) or D_out rows
    # (in Cell1, Cell2, ...) are bottom input weights, and the rest D_out rows
    # are state weights, i.e. *inputs are before states in weight matrix*
    #
    # The gate order in 4*D_out are i, j, f, o, where
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    #
    # Other details in tensorflow/python/ops/rnn_cell.py
    with tf.variable_scope(name):
        # the basic LSTM cell
        lstm_cell_fw = tf.contrib.rnn.LSTMCell(output_dim, forget_bias, initializer=initializer)
        lstm_cell_bw = tf.contrib.rnn.LSTMCell(output_dim, forget_bias, initializer=initializer)
        # Apply dropout if specified.
        if apply_dropout and keep_prob < 1:
            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(
                lstm_cell_fw, output_keep_prob=keep_prob)
            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(
                lstm_cell_bw, output_keep_prob=keep_prob)

        cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * num_layers)
        cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * num_layers)

        # Initialize cell state from zero.
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
        # Fix batch_size issue when batch_size == 1
        # state_shape = initial_state.get_shape().as_list()
        # state_shape[0] = batch_size
        # initial_state.set_shape(state_shape)

        # Split along time dimension and flatten each component.
        # `inputs` is a list.
        inputs = [tf.reshape(input_, [batch_size, -1])
            for input_ in tf.split(seq_bottom, num_steps, 0)]
        # Add constant input to each time step.
        if not const_bottom is None:
            # Flatten const_bottom into shape [N, D_const] and concatenate.
            const_input_ = tf.reshape(const_bottom, [batch_size, -1])
            inputs = [tf.concat(0, [input_, const_input_])
                for input_ in inputs]

        # Create the Recurrent Network and collect `outputs` and `states`
        outputs, states_fw, states_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, 
            inputs, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)
        if concat_output:
            # Concat the outputs into [T, N, D_out].
            outputs = tf.reshape(tf.concat(0, outputs),
                                [num_steps, batch_size, output_dim])
    return outputs, states_fw, states_bw


