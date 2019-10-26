from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.contrib import rnn
from commons.common_ops import create_weight

def get_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def _conv_opt(inputs, window_size, out_filters, separable=False, ch_mul=1, is_mask=True):
    inp_d = inputs.get_shape()[1].value
    inp_l = inputs.get_shape()[2].value

    if is_mask:
      inputs = tf.transpose(inputs, [0, 2, 1])  # [batch_size, len, dim]
      valid_length = get_length(inputs)
      mask = tf.sequence_mask(valid_length, inp_l)
      mask = tf.expand_dims(mask, 1)
      mask = tf.tile(mask, [1, inp_d, 1])
      #mask = tf.expand_dims(mask, 2)  # [batch_size, dim, 1, len]
      inputs = tf.transpose(inputs, [0, 2, 1])  # [batch_size, len, dim]
      inputs = tf.reshape(inputs, [-1, inp_d, inp_l])
      inputs = tf.where(mask, inputs, tf.zeros_like(inputs))
      #print("is mask", inputs.shape)

    if separable == True:
      w_depth = create_weight(
          "w_depth", [window_size, out_filters, ch_mul])
      w_point = create_weight("w_point", [1, out_filters * ch_mul, out_filters])
      out = tf.nn.separable_conv1d(inputs, w_depth, w_point, strides=1,
                                   padding="SAME", data_format='NCW')
    else:
      w = create_weight("w", [window_size, inp_d, out_filters])
      #print("w_weight", w.shape, inputs.shape)
      out = tf.nn.conv1d(inputs, w, 1, "SAME", data_format='NCW')
      #print("out", out.shape)

    if is_mask:
      mask = tf.sequence_mask(valid_length, inp_l)
      mask = tf.expand_dims(mask, 1)
      mask = tf.tile(mask, [1, out_filters, 1])
      #mask = tf.expand_dims(mask, 2)  # [batch_size, dim, 1, len]
      out = tf.where(mask, out, tf.zeros_like(out))
      out = tf.reshape(out, [-1, out_filters, inp_l])

    return out

def conv_op(inputs, filter_size, is_training, count, out_filters,
            ch_mul=1, start_idx=None, separable=False, is_mask=True):
    """
    Args:
        start_idx: where to start taking the output channels. if None, assuming
            fixed_arc mode
        count: how many output_channels to take.
    """

    x = inputs

    with tf.variable_scope("out_conv_{0}".format(filter_size)):
        if start_idx is None:
            if separable:
                x = _conv_opt(x, filter_size, out_filters, separable=True, ch_mul=ch_mul, is_mask=is_mask)
                x = batch_norm(x, is_training)
            else:
                x = _conv_opt(x, filter_size, out_filters,  is_mask=is_mask)
                x = batch_norm(x, is_training)
        else:
            if separable:
                x = _conv_opt(x, filter_size, out_filters, separable=True, ch_mul=ch_mul, is_mask=is_mask)
                mask = tf.range(0, out_filters, dtype=tf.int32)
                mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
                x = batch_norm_with_mask(
                    x, is_training, mask, out_filters)
            else:
                x = _conv_opt(x, filter_size, count, is_mask=is_mask)
                mask = tf.range(0, out_filters, dtype=tf.int32)
                mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
                x = batch_norm_with_mask(
                    x, is_training, mask, out_filters)
        x = tf.nn.relu(x)
    return x

def pool_op(inputs, is_training, count, out_filters, avg_or_max, start_idx=None):
    """
    Args:
        start_idx: where to start taking the output channels. if None, assuming
            fixed_arc mode
        count: how many output_channels to take.
    """

    inp_d = inputs.get_shape()[1].value

    with tf.variable_scope("conv_1"):
        w = create_weight("w", [1, inp_d, out_filters])
        x = tf.nn.conv1d(inputs, w, 1, "SAME", data_format='NCW')
        x = batch_norm(x, is_training)
        x = tf.nn.relu(x)

    with tf.variable_scope("pool"):
        actual_data_format = "channels_first"

        if avg_or_max == "avg":
            x = tf.layers.average_pooling1d(
                x, 3, 1, "SAME", data_format=actual_data_format)
        elif avg_or_max == "max":
            x = tf.layers.max_pooling1d(
                x, 3, 1, "SAME", data_format=actual_data_format)
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))

        if start_idx is not None:
            x = x[:, start_idx: start_idx+count, :]

    return x


def batch_norm(x, is_training, name="bn", decay=0.9, epsilon=1e-5):

    inp_d = x.get_shape()[1].value
    inp_l = x.get_shape()[2].value
    x = tf.expand_dims(x, 2)
    data_format = "NCHW"

    shape = [x.get_shape()[1]]

    with tf.variable_scope(name, reuse=None if is_training else True):
        offset = tf.get_variable(
            "offset", shape,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        scale = tf.get_variable(
            "scale", shape,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        moving_mean = tf.get_variable(
            "moving_mean", shape, trainable=False,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        moving_variance = tf.get_variable(
            "moving_variance", shape, trainable=False,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        if is_training:
            x, mean, variance = tf.nn.fused_batch_norm(
                x, scale, offset, epsilon=epsilon, data_format=data_format,
                is_training=True)
            update_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)
        else:
            x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                             variance=moving_variance,
                                             epsilon=epsilon, data_format=data_format,
                                             is_training=False)
    x = tf.reshape(x, [-1, inp_d, inp_l])

    return x


def batch_norm_with_mask(x, is_training, mask, num_channels, name="bn",
                         decay=0.9, epsilon=1e-3):

    inp_d = x.get_shape()[1].value
    inp_l = x.get_shape()[2].value
    x = tf.expand_dims(x, 2)
    data_format = "NCHW"

    shape = [num_channels]
    indices = tf.where(mask)
    indices = tf.to_int32(indices)
    indices = tf.reshape(indices, [-1])

    with tf.variable_scope(name, reuse=None if is_training else True):
        offset = tf.get_variable(
            "offset", shape,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        scale = tf.get_variable(
            "scale", shape,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        offset = tf.boolean_mask(offset, mask)
        scale = tf.boolean_mask(scale, mask)

        moving_mean = tf.get_variable(
            "moving_mean", shape, trainable=False,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        moving_variance = tf.get_variable(
            "moving_variance", shape, trainable=False,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        if is_training:
            x, mean, variance = tf.nn.fused_batch_norm(
                x, scale, offset, epsilon=epsilon, data_format=data_format,
                is_training=True)
            mean = (1.0 - decay) * (tf.boolean_mask(moving_mean, mask) - mean)
            variance = (1.0 - decay) * \
                (tf.boolean_mask(moving_variance, mask) - variance)
            update_mean = tf.scatter_sub(
                moving_mean, indices, mean, use_locking=True)
            update_variance = tf.scatter_sub(
                moving_variance, indices, variance, use_locking=True)
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)
        else:
            masked_moving_mean = tf.boolean_mask(moving_mean, mask)
            masked_moving_variance = tf.boolean_mask(moving_variance, mask)
            x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                             mean=masked_moving_mean,
                                             variance=masked_moving_variance,
                                             epsilon=epsilon, data_format=data_format,
                                             is_training=False)
    x = tf.reshape(x, [-1, inp_d, inp_l])
    return x

def recur_op(inputs, is_training, count, out_filters, start_idx=None, is_height=False,
             lstm_x_keep_prob=1.0, lstm_h_keep_prob=1.0, lstm_o_keep_prob=1.0, var_rec=False):
    """
        Args:
          start_idx: where to start taking the output channels. if None, assuming
            fixed_arc mode
          count: how many output_channels to take.
        """

    x = inputs

    batch_size = tf.shape(x)[0]
    inp_d = inputs.get_shape()[1].value
    inp_l = inputs.get_shape()[2].value

    with tf.variable_scope("recur_{0}".format(is_height)):
        if start_idx is None:
            #x = tf.transpose(x, [0, 2, 3, 1])
            if is_height == False:
                #x = tf.transpose(x, [0, 2, 1, 3])
                x = tf.reshape(x, [batch_size, inp_l, inp_d])
                rnn_out_filters = 1
            else:
                x = tf.reshape(x, [batch_size, 1, inp_l * inp_d])
                rnn_out_filters = inp_l

            with tf.variable_scope("cell_fw"):
                cell_fw = rnn.GRUCell(rnn_out_filters * inp_d, reuse=tf.AUTO_REUSE)
            with tf.variable_scope("cell_bw"):
                cell_bw = rnn.GRUCell(rnn_out_filters * inp_d, reuse=tf.AUTO_REUSE)
            length = get_length(x)
            if is_training:
                cell_fw = rnn.DropoutWrapper(cell_fw, input_keep_prob=lstm_x_keep_prob,
                                             state_keep_prob=lstm_h_keep_prob,
                                             output_keep_prob=lstm_o_keep_prob,
                                             variational_recurrent=var_rec, dtype=tf.float32,
                                             input_size=x.get_shape()[-1])
                cell_bw = rnn.DropoutWrapper(cell_bw, input_keep_prob=lstm_x_keep_prob,
                                             state_keep_prob=lstm_h_keep_prob,
                                             output_keep_prob=lstm_o_keep_prob,
                                             variational_recurrent=var_rec, dtype=tf.float32,
                                             input_size=x.get_shape()[-1])
            (outputs_fw, outputs_bw), states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, x, dtype=tf.float32, sequence_length=length)
            outputs = tf.add_n([outputs_fw, outputs_bw])

            x = tf.reshape(outputs, [batch_size, inp_d, inp_l])

        else:
            if is_height == False:
                #x = tf.transpose(x, [0, 2, 1, 3])
                x = tf.reshape(x, [batch_size, inp_l, inp_d]) # (?, max_len, dim)
                rnn_out_filters = 1
            else:
                x = tf.reshape(x, [batch_size, 1, inp_l * inp_d])
                rnn_out_filters = inp_l

            with tf.variable_scope("cell_fw"):
                cell_fw = rnn.GRUCell(rnn_out_filters * inp_d, reuse=tf.AUTO_REUSE)
            with tf.variable_scope("cell_bw"):
                cell_bw = rnn.GRUCell(rnn_out_filters * inp_d, reuse=tf.AUTO_REUSE)
            length = get_length(x)
            if is_training:
                cell_fw = rnn.DropoutWrapper(cell_fw, input_keep_prob=lstm_x_keep_prob,
                                             state_keep_prob=lstm_h_keep_prob,
                                             output_keep_prob=lstm_o_keep_prob,
                                             variational_recurrent=var_rec, dtype=tf.float32,
                                             input_size=x.get_shape()[-1])
                cell_bw = rnn.DropoutWrapper(cell_bw, input_keep_prob=lstm_x_keep_prob,
                                             state_keep_prob=lstm_h_keep_prob,
                                             output_keep_prob=lstm_o_keep_prob,
                                             variational_recurrent=var_rec, dtype=tf.float32,
                                             input_size=x.get_shape()[-1])
            (outputs_fw, outputs_bw), states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, x, dtype=tf.float32,
                sequence_length=length)
            outputs = tf.add_n([outputs_fw, outputs_bw])

            x = tf.reshape(outputs, [batch_size, inp_d, inp_l])

            mask = tf.range(0, out_filters, dtype=tf.int32)
            mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)

    return x

def multihead_attention(queries,
                        keys,
                        pos_embedding,
                        field_embedding,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        positional_encoding=False,
                        do_field_embedding=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
      # Set the fall back option for num_units
      if num_units is None:
        num_units = queries.get_shape().as_list[-1]

      # Add positional and field embedding
      m_queries = queries
      m_keys = keys
      if positional_encoding:
        cur_pos_embedding = tf.reduce_sum(pos_embedding, axis=2)
        cur_pos_embedding = tf.transpose(cur_pos_embedding, [0, 2, 1])
        m_queries += cur_pos_embedding
        m_keys += cur_pos_embedding
      if do_field_embedding:
        cur_field_embedding = tf.reduce_sum(field_embedding, axis=2)
        cur_field_embedding = tf.transpose(cur_field_embedding, [0, 2, 1])
        m_queries += cur_field_embedding
        m_keys += cur_pos_embedding

      # Linear projections
      Q = tf.layers.dense(m_queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
      K = tf.layers.dense(m_keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
      V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

      # Split and concat
      Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
      K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
      V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

      # Multiplication
      outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

      # Scale
      outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

      # Key Masking
      key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
      key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
      key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

      paddings = tf.ones_like(outputs)*(-2**32+1)
      outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

      # Causality = Future blinding
      if causality:
        diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
        tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks)*(-2**32+1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

      # Activation
      outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

      # Query Masking
      query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
      query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
      query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
      outputs *= query_masks # broadcasting. (N, T_q, C)

      # Dropouts
      outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

      # Weighted sum
      outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

      # Restore shape
      outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

    return outputs

def _attention_opt(inputs, pos_embedding, field_embedding,out_filters, head_num, is_training, keep_ratio,
                   positional_encoding=False, do_field_embedding=False):

    inp_d = inputs.get_shape()[1].value
    inp_l = inputs.get_shape()[2].value

    print("dim, len", inp_d, inp_l)  # dim, len
    #put channel to the last dim
    inputs = tf.reshape(inputs, [-1, inp_l, inp_d])

    with tf.variable_scope("attention"):
      out = multihead_attention(queries=inputs,
                                   keys=inputs,
                                   pos_embedding=pos_embedding,
                                   field_embedding=field_embedding,
                                   num_units=inp_d,
                                   num_heads=head_num,
                                   dropout_rate=1-keep_ratio,
                                   is_training=is_training,
                                   causality=False,
                                   positional_encoding=positional_encoding,
                                   do_field_embedding=do_field_embedding)

    out = tf.reshape(out, [-1, inp_d, inp_l])

    return out

def attention_op(inputs, pos_embedding, field_embedding, is_training, count, out_filters,
                        start_idx=None, positional_encoding=False, attention_keep_prob=1.0,
                 do_field_embedding=False):

    out = inputs
    with tf.variable_scope("out_attention"):
      out = _attention_opt(out, pos_embedding, field_embedding, out_filters,
                                8, is_training, attention_keep_prob, positional_encoding=positional_encoding,
                           do_field_embedding=do_field_embedding)

      if start_idx is None:
        out = batch_norm(out, is_training)
      else:
        mask = tf.range(0, out_filters, dtype=tf.int32)
        mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
        out = batch_norm_with_mask(
          out, is_training, mask, out_filters)
    return out




