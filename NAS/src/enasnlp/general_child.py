from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import random
import numpy as np
import tensorflow as tf

from src.enasnlp.models import Model
from src.utils import count_model_params, get_train_ops
from src.common_ops import create_weight, batch_norm, conv_op, pool_op, global_avg_pool, recur_op, attention_op, \
  multihead_attention, _conv_opt

class GeneralChild(Model):
  def __init__(self,
               images_train,
               bow_images_train,
               labels_train,
               datasets_train,
               images,
               labels,
               datasets,
               embedding,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
               num_branches=6,
               out_filters=24,
               keep_prob=1.0,
               lstm_x_keep_prob=1.0,
               lstm_h_keep_prob=1.0,
               lstm_o_keep_prob=1.0,
               embed_keep_prob=1.0,
               attention_keep_prob=1.0,
               var_rec=False,
               batch_size=32,
               clip_mode=None,
               grad_bound=None,
               lr_warmup_steps=100,
               lr_model_d=300,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               lr_decay_scheme="exponential",
               lr_decay_epoch_multiplier=1.0,
               lr_max=None,
               lr_min=None,
               lr_T_0=None,
               lr_T_mul=None,
               optim_algo=None,
               data_format="NHWC",
               dataset="sst",
               multi_path=False,
               positional_encoding=False,
               input_positional_encoding=False,
               is_sinusolid=False,
               embedding_model="word2vec",
               skip_concat=False,
               all_layer_output=False,
               num_last_layer_output=0,
               output_linear_combine=False,
               is_debug=False,
               is_mask=False,
               is_output_attention=False,
               field_embedding=False,
               input_field_embedding=False,
               sliding_window=False,
               name="child",
               pool_step=3,
               class_num=5,
               mode="subgraph",
               *args,
               **kwargs
              ):
    """
    """

    super(self.__class__, self).__init__(
      images_train,
      bow_images_train,
      labels_train,
      datasets_train,
      images,
      labels,
      datasets,
      batch_size=batch_size,
      clip_mode=clip_mode,
      grad_bound=grad_bound,
      l2_reg=l2_reg,
      lr_init=lr_init,
      lr_dec_start=lr_dec_start,
      lr_dec_every=lr_dec_every,
      lr_dec_rate=lr_dec_rate,
      keep_prob=keep_prob,
      embed_keep_prob=embed_keep_prob,
      lstm_x_keep_prob=lstm_x_keep_prob,
      lstm_h_keep_prob=lstm_h_keep_prob,
      lstm_o_keep_prob=lstm_o_keep_prob,
      attention_keep_prob=attention_keep_prob,
      var_rec=var_rec,
      optim_algo=optim_algo,
      data_format=data_format,
      dataset=dataset,
      name=name)

    self.embedding = embedding
    self.embedding_model = embedding_model
    self.field_embedding = field_embedding
    self.input_field_embedding = input_field_embedding
    self.sliding_window = sliding_window
    self.skip_concat = skip_concat
    self.all_layer_output = all_layer_output
    self.num_last_layer_output = max(num_last_layer_output, 0)
    self.output_linear_combine = output_linear_combine
    self.is_debug = is_debug
    self.is_mask = is_mask
    self.is_output_attention = is_output_attention
    self.multi_path = multi_path
    self.positional_encoding = positional_encoding
    self.input_positional_encoding = input_positional_encoding
    self.is_sinusolid = is_sinusolid
    self.lr_decay_scheme = lr_decay_scheme
    self.lr_decay_epoch_multiplier = lr_decay_epoch_multiplier
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul
    self.lr_warmup_val = None
    self.lr_warmup_steps = lr_warmup_steps
    self.lr_model_d = lr_model_d
    self.out_filters = out_filters * out_filters_scale
    self.num_layers = num_layers
    self.pool_step = pool_step
    self.class_num = class_num
    self.mode = mode

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // self.pool_step
    self.pool_layers = []
    for i in range(1, self.pool_step):
      self.pool_layers.append(i * pool_distance - 1)
    self.w_emb = tf.get_variable("w_out", [10000, self.out_filters])

  def _get_strides(self, stride):
    return [1, 1, 1, stride]

  def _factorized_reduction(self, x, out_filters, stride, is_training):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      with tf.variable_scope("path_conv"):
        x = _conv_opt(x, 1, out_filters)
        x = batch_norm(x, is_training, data_format=self.data_format)
        return x

    actual_data_format = "channels_first"  # only support NCHW
    stride_spec = self._get_strides(stride)
    # Skip path 1
    path1 = tf.nn.max_pool(
        x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)

    with tf.variable_scope("path1_conv"):
      path1 = _conv_opt(path1, 1, out_filters // 2)

    # Skip path 2
    # First pad with 0"s on the right and bottom, then shift the filter to
    # include those 0"s that were added.
    if self.data_format == "NHWC":
      pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
      path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
      inp_c = path2.get_shape()[1].value
      if inp_c > 1:
        concat_axis = 3
      else:
        concat_axis = 1
    else:
      pad_arr = [[0, 0], [0, 0], [0, 0], [0, 1]]
      path2 = tf.pad(x, pad_arr)[:, :, :, 1:]
      inp_c = path2.get_shape()[1].value
      if inp_c > 1:
        concat_axis = 1
      else:
        concat_axis = 2

    path2 = tf.nn.max_pool(
        path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
    with tf.variable_scope("path2_conv"):
      path2 = _conv_opt(path2, 1, out_filters // 2)

    # Concat and apply BN
    final_path = tf.concat(values=[path1, path2], axis=concat_axis)
    final_path = batch_norm(final_path, is_training,
                            data_format=self.data_format)

    return final_path

  def _positional_encoding(self, inputs, batch_size, is_training, num_units, zero_pad=True,
                           scale=True, scope="positional_encoding", reuse=None, mode="train"):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N = batch_size
    T = inputs.get_shape()[3]
    with tf.variable_scope(scope, reuse=reuse):
      position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

      # First part of the PE function: sin and cos argument
      position_enc = np.array([
          [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
          for pos in range(T)])

      # Second part, apply the cosine to even columns and sin to odds.
      position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
      position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

      # Convert to a tensor
      lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
      if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),lookup_table[1:, :]), 0)
      outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
      if scale:
        outputs = outputs * num_units**0.5

    return outputs


  def _embedding(self, inputs, vocab_size, num_units, zero_pad=True,
              scale=True, scope="embedding", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = create_weight("lookup_table",
                                     trainable=True,
                                     shape=[vocab_size, num_units],
                                     initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
          lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
          outputs = outputs * (num_units ** 0.5)

    return outputs

  def _to_sliding_window(self, images, batch_size, size, step):
    result_images, sliding_windows = [], []
    self.cur_batch_size = 0
    max_len = images.get_shape()[1]
    for i in range(0, batch_size):
      start, count = 0, 0
      while start + size <= max_len:
        image = tf.slice(images, [i, start], [1, size])
        image = tf.reshape(image, [size])
        result_images.append(image)
        start += step
        count += 1
        self.cur_batch_size += 1
      sliding_windows.append(count)
    result_images = tf.convert_to_tensor(result_images)
    return result_images, sliding_windows

  def _from_sliding_window(self, x, batch_size, sliding_windows):
    result_images, images = [], []
    index = 0
    print("cur_batch_size: {0}".format(self.cur_batch_size))
    print("sliding_windows: {0}".format(sliding_windows))
    for i in range(0, self.cur_batch_size):
      images.append(x[i])
      if len(images) == sliding_windows[index]:
        image = tf.add_n(images)
        result_images.append(image)
        images = []
        index += 1
    result_images = tf.convert_to_tensor(result_images)
    print("result_images: {0}".format(result_images))
    return result_images

  def _model(self, images, bow_images, datasets, is_training,
             reuse=False, mode="train"):
    is_debug = self.is_debug and is_training
    with tf.variable_scope(self.name, reuse=reuse):
      layers = []
      final_flags = []
      pre_idxs = []
      if is_training:
        self.valid_lengths = []
      with tf.variable_scope('embed'):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)
        if self.embedding_model == "none":
          embedding = create_weight("w", shape=self.embedding["none"].shape, trainable=True,
                                  initializer=tf.truncated_normal_initializer, regularizer=regularizer)
        elif self.embedding_model == "glove":
          embedding = create_weight("w", shape=None, trainable=True,
                                  initializer=self.embedding["glove"], regularizer=regularizer)
        elif self.embedding_model == "word2vec":
          embedding = create_weight("w", shape=None, trainable=True,
                                  initializer=self.embedding["word2vec"], regularizer=regularizer)
        elif self.embedding_model == "all":
          embedding_glove = create_weight("w_glove", shape=None, trainable=True,
                                  initializer=self.embedding["glove"], regularizer=regularizer)
          print("embedding_glove: {0}".format(embedding_glove.get_shape()))
          embedding_word2vec = create_weight("w_word2vec", shape=None, trainable=True,
                                  initializer=self.embedding["word2vec"], regularizer=regularizer)
          print("embedding_word2vec: {0}".format(embedding_word2vec.get_shape()))
          embedding = tf.concat([embedding_glove, embedding_word2vec], axis=0)
          print("join embedding: {0}".format(embedding.get_shape()))
        field_embedding = create_weight("w_field", shape=self.embedding["field"].shape, trainable=True,
                                  initializer=tf.truncated_normal_initializer, regularizer=regularizer)

        self.final_embedding = embedding
        print("embedding: {0}".format(embedding))
        print("images: {0}".format(images))
        print("bow_images: {0}".format(bow_images))

        if is_training or mode =="valid":
          batch_size = self.batch_size
        else:
          batch_size = self.eval_batch_size

        if self.sliding_window:
          images, sliding_windows = self._to_sliding_window(images, batch_size, size=64, step=32)
          bow_images, _ = self._to_sliding_window(bow_images, batch_size, size=64, step=32)
          print("images after sliding window: {0}".format(images))

        if is_training:
          embedding = tf.nn.dropout(embedding, keep_prob=self.embed_keep_prob)

        images = tf.nn.embedding_lookup(embedding, images, max_norm=None)
        field_embedding = tf.nn.embedding_lookup(field_embedding, bow_images, max_norm=None)
        if self.input_field_embedding:
          images = tf.add_n([images, field_embedding])
        images = tf.transpose(images, [0, 2, 1])
        inp_c = images.shape[1]
        inp_w = images.shape[2]
        images = tf.reshape(images, [-1, inp_c, 1, inp_w])
        field_embedding = tf.transpose(field_embedding, [0, 2, 1])
        field_embedding = tf.reshape(field_embedding, [-1, inp_c, 1, inp_w])

      x = images
      pos_batch_size = 1
      # initialize pos_embedding for transformer
      if self.input_positional_encoding:
        out_filters = 300
      else:
        out_filters = self.out_filters
      if self.is_sinusolid:
        pos_embedding = self._positional_encoding(x, pos_batch_size, is_training, num_units=out_filters,
                                zero_pad=False, scale=False, scope="enc_pe")
      else:
        pos_embedding = self._embedding(tf.tile(tf.expand_dims(tf.range(inp_w), 0), [pos_batch_size, 1]),
                                vocab_size=inp_w,
                                num_units=out_filters,
                                reuse=tf.AUTO_REUSE,
                                zero_pad=True,
                                scale=False,
                                scope="enc_pe")
      print("pos embedding: {0}".format(pos_embedding))
      pos_embedding = tf.transpose(pos_embedding, [0, 2, 1])
      pos_embedding = tf.expand_dims(pos_embedding, axis=2)
      print("pos embedding: {0}".format(pos_embedding))
      if self.input_positional_encoding:
        x += pos_embedding

      out_filters = self.out_filters
      with tf.variable_scope("init_conv"):  # adjust out_filter dimension
        x = _conv_opt(x, 1, self.out_filters)
        x = batch_norm(x, is_training, data_format=self.data_format)

        layers.append(x)

      # sveral operations for nni
      def add_fixed_pooling_layer(layer_id, layers, out_filters, is_training, pos_embedding, field_embedding):
        '''Add a fixed pooling layer every four layers'''
        with tf.variable_scope("pos_embed_pool_{0}".format(layer_id)):
          pos_embedding = self._factorized_reduction(
            pos_embedding, out_filters, 2, is_training)

        with tf.variable_scope("field_embed_pool_{0}".format(layer_id)):
          field_embedding = self._factorized_reduction(
            field_embedding, out_filters, 2, is_training)

        #out_filters *= 2
        with tf.variable_scope("pool_at_{0}".format(layer_id)):
          pooled_layers = []
          for i, layer in enumerate(layers):
            #print("pooling_layer", i, layer)
            with tf.variable_scope("from_{0}".format(i)):
              x = self._factorized_reduction(
                layer, out_filters, 2, is_training)
              #print("after x ", x)
            pooled_layers.append(x)

          layers = pooled_layers

          return layers, out_filters

      def post_process_out(inputs, out):
        '''Form skip connection and perform batch norm'''
        optional_inputs = inputs[1]
        print("post_process_out::", inputs, optional_inputs)
        with tf.variable_scope(get_layer_id()):
          with tf.variable_scope("skip"):
            #print("layers",layers)
            inputs = layers[-1]
            if self.data_format == "NHWC":
              inp_h = inputs.get_shape()[1].value
              inp_w = inputs.get_shape()[2].value
              inp_c = inputs.get_shape()[3].value
              out.set_shape([None, inp_h, inp_w, out_filters])
            elif self.data_format == "NCHW":
              inp_c = inputs.get_shape()[1].value
              inp_h = inputs.get_shape()[2].value
              inp_w = inputs.get_shape()[3].value
              out.set_shape([None, out_filters, inp_h, inp_w])
            try:
              out = tf.add_n(
                [out, tf.reduce_sum(optional_inputs, axis=0)])
            except Exception as e:
              print(e)
            out = batch_norm(out, is_training,
                             data_format=self.data_format)
        layers.append(out)
        return out

      global layer_id
      layer_id = -1

      def get_layer_id():
        global layer_id
        layer_id += 1
        return 'layer_' + str(layer_id)

      size = [1, 3, 5, 7]
      separables = [False, False, False, False]

      def conv(inputs, size, separable=False):
        # res_layers is pre_layers that are chosen to form skip connection
        # layers[-1] is always the latest input
        with tf.variable_scope(get_layer_id()):
          with tf.variable_scope('conv_' + str(size) + ('_separable' if separable else '')):
            #print("conv_inputs::", inputs)
            dealed_inputs = tf.reduce_sum(inputs[1], axis=0)
            #print("dealed_inputs::", dealed_inputs)
            out = conv_op(
              dealed_inputs, size, is_training, out_filters, out_filters, start_idx=None,
              separable=separable)
        #layers.append(out)
        return out

      def pool(inputs, ptype):
        assert ptype in ['avg', 'max'], "pooling type must be avg or max"

        with tf.variable_scope(get_layer_id()):
          with tf.variable_scope('pooling_' + str(ptype)):
            #print("pool_inputs::", inputs)
            dealed_inputs = tf.reduce_sum(inputs[1], axis=0)
            #print("dealed_inputs::", dealed_inputs)
            out = pool_op(
              dealed_inputs, is_training, out_filters, out_filters, ptype, self.data_format, start_idx=None)
        #layers.append(out)
        return out

      def rnn(inputs):

        with tf.variable_scope(get_layer_id()):
          with tf.variable_scope('branch_6'):
            #print("rnn_inputs::", inputs)
            dealed_inputs = tf.reduce_sum(inputs[1], axis=0)
            #print("dealed_inputs::", dealed_inputs)
            out = recur_op(
              dealed_inputs, is_training, out_filters, out_filters, start_idx=0, data_format=self.data_format,
             lstm_x_keep_prob=self.lstm_x_keep_prob, lstm_h_keep_prob=self.lstm_h_keep_prob, lstm_o_keep_prob=self.lstm_o_keep_prob, var_rec=self.var_rec)
        #layers.append(out)
        return out

      def attention(inputs):

        with tf.variable_scope(get_layer_id()):
          with tf.variable_scope('branch_7'):
            #print("attention_inputs::", inputs)
            dealed_inputs = tf.reduce_sum(inputs[1], axis=0)
            #print("dealed_inputs::", dealed_inputs)
            out = attention_op(
              dealed_inputs, pos_embedding, field_embedding, is_training, out_filters, out_filters, start_idx=0,
              positional_encoding=self.positional_encoding, attention_keep_prob=self.attention_keep_prob, data_format=self.data_format,
            do_field_embedding=self.field_embedding)
        #layers.append(out)
        return out

      def final_process(inputs):
        with tf.variable_scope(get_layer_id()):
          with tf.variable_scope('final_out'):
            print("final_inputs::", inputs)
            dealed_inputs = tf.reduce_mean(inputs[1], axis=0)
            print("dealed_inputs::", dealed_inputs)
            out = dealed_inputs
            #out = tf.reduce_mean(inputs[1], axis=0)
            print("final_out::", inputs, out)
        layers.append(out)
        return out

      """@nni.mutable_layers(
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x],
          optional_input_size: 1,
          layer_output: layer_0_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_0_out_0)],
          optional_inputs: [],
          optional_input_size: 1,
          layer_output: layer_0_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out],
          optional_input_size: 1,
          layer_output: layer_1_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_1_out_0)],
          optional_inputs: [layer_0_out],
          optional_input_size: 1,
          layer_output: layer_1_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out],
          optional_input_size: 1,
          layer_output: layer_2_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_2_out_0)],
          optional_inputs: [layer_0_out, layer_1_out],
          optional_input_size: 1,
          layer_output: layer_2_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out],
          optional_input_size: 1,
          layer_output: layer_3_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_3_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out],
          optional_input_size: 1,
          layer_output: layer_3_out
      }
      )"""
      layers, out_filters = add_fixed_pooling_layer(
        3, layers, out_filters, is_training, pos_embedding, field_embedding)
      x, layer_0_out, layer_1_out, layer_2_out, layer_3_out = layers[-5:]
      print("layer_out", x, layer_0_out, layer_1_out, layer_2_out, layer_3_out)

      """@nni.mutable_layers(
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out],
          optional_input_size: 1,
          layer_output: layer_4_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_4_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out],
          optional_input_size: 1,
          layer_output: layer_4_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out],
          optional_input_size: 1,
          layer_output: layer_5_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_5_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out],
          optional_input_size: 1,
          layer_output: layer_5_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out],
          optional_input_size: 1,
          layer_output: layer_6_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_6_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out],
          optional_input_size: 1,
          layer_output: layer_6_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out],
          optional_input_size: 1,
          layer_output: layer_7_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_7_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out],
          optional_input_size: 1,
          layer_output: layer_7_out
      }
      )"""
      layers, out_filters = add_fixed_pooling_layer(
        7, layers, out_filters, is_training, pos_embedding, field_embedding)
      x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out = layers[
                                                                                                               -9:]
      """@nni.mutable_layers(
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out],
          optional_input_size: 1,
          layer_output: layer_8_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_8_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out],
          optional_input_size: 1,
          layer_output: layer_8_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out],
          optional_input_size: 1,
          layer_output: layer_9_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_9_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out],
          optional_input_size: 1,
          layer_output: layer_9_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out],
          optional_input_size: 1,
          layer_output: layer_10_out_0
      },
      {
          layer_choice: [post_process_out(out=layer_10_out_0)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out],
          optional_input_size: 1,
          layer_output: layer_10_out
      },
      {
          layer_choice: [conv(size=1), conv(size=3), conv(size=5), conv(size=7), pool(ptype='avg'), pool(ptype='max'), rnn(), attention()],
          optional_inputs: [x, layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_10_out],
          optional_input_size: 1,
          layer_output: layer_11_out_1
      },
      {
          layer_choice: [post_process_out(out=layer_11_out_1)],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_10_out],
          optional_input_size: 1,
          layer_output: layer_11_out
      },
      {
          layer_choice: [final_process()],
          optional_inputs: [layer_0_out, layer_1_out, layer_2_out, layer_3_out, layer_4_out, layer_5_out, layer_6_out, layer_7_out, layer_8_out, layer_9_out, layer_10_out, layer_11_out],
          optional_input_size: 1,
          layer_output: final_out
      }
      )"""


      print("len_layers: ", len(layers))
      x = final_out

      if self.sliding_window:
        x = self._from_sliding_window(x, batch_size, sliding_windows)

      class_num = self.class_num
      with tf.variable_scope("fc"):
        if not self.is_output_attention:
          x = global_avg_pool(x, data_format=self.data_format)
        else:
          batch_size = x.get_shape()[0].value
          inp_c = x.get_shape()[1].value
          inp_h = x.get_shape()[2].value
          inp_w = x.get_shape()[3].value
          final_attention_query = create_weight("query", shape=[1, 1, inp_c], trainable=True,
                                    initializer=tf.truncated_normal_initializer, regularizer=regularizer)
          if is_training or mode == "valid":
            batch_size = self.batch_size
          else:
            batch_size = self.eval_batch_size
          final_attention_query = tf.tile(final_attention_query, [batch_size, 1, 1])
          print("final_attention_query: {0}".format(final_attention_query))

          #put channel to the last dim
          x = tf.transpose(x, [0, 2, 3, 1])        #1, 24, 32
          x = tf.reshape(x, [-1, inp_h*inp_w, inp_c])
          print("x: {0}".format(x))
          x = multihead_attention(queries=final_attention_query,
                                   keys=x,
                                   pos_embedding=pos_embedding,
                                   field_embedding=field_embedding,
                                   num_units=inp_c,
                                   num_heads=8,
                                   dropout_rate=0,
                                   is_training=is_training,
                                   causality=False,
                                   positional_encoding=self.positional_encoding)
          print("x: {0}".format(x))
          x = tf.reshape(x, [-1, 1, inp_c])
          x = tf.reduce_sum(x, axis=1)
          print("x: {0}".format(x))
        if is_training:
          x = tf.nn.dropout(x, self.keep_prob)
        x = tf.layers.dense(x, units=class_num)

    return x

  # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    x_train = self.x_train
    logits = self._model(x_train, self.x_bow_train, self.d_train,
                        is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print("Model has {} params".format(self.num_vars))

    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")

    if self.lr_decay_scheme == "auto":
      self.step_value = tf.placeholder(tf.int32, shape=(), name="step_value")
      self.assign_global_step = self.global_step.assign(self.step_value)

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_warmup_val=self.lr_warmup_val,
      lr_warmup_steps=self.lr_warmup_steps,
      lr_model_d=self.lr_model_d,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      lr_decay_scheme=self.lr_decay_scheme,
      lr_max=self.lr_max,
      lr_min=self.lr_min,
      lr_T_0=self.lr_T_0,
      lr_T_mul=self.lr_T_mul,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo,
      lr_decay_epoch_multiplier=self.lr_decay_epoch_multiplier)

  # override
  def _build_valid(self):
    if self.x_valid is not None:
      print("-" * 80)
      print("Build valid graph")
      x_valid = self.x_valid
      logits = self._model(x_valid, self.x_bow_valid, self.d_valid, False, reuse=True, mode="valid")
      self.valid_preds = tf.argmax(logits, axis=1)
      self.valid_preds = tf.to_int32(self.valid_preds)
      self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
      self.valid_acc = tf.to_int32(self.valid_acc)
      self.valid_acc = tf.reduce_sum(self.valid_acc)

  # override
  def _build_test(self):
    print("-" * 80)
    print("Build test graph")
    x_test = self.x_test
    logits = self._model(x_test, self.x_bow_test, self.d_test, False, reuse=True)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

  # override
  def _build_valid_rl(self, shuffle=False):
      print("-" * 80)
      print("Build valid graph on shuffled data")
      with tf.device("/cpu:0"):
        x_valid_shuffle, x_bow_valid_shuffle, y_valid_shuffle, d_valid_shuffle = tf.train.shuffle_batch(
          [self.images["valid_original"], self.images["valid_bow_ids_original"],
           self.labels["valid_original"], self.datasets["valid_original"]],
          batch_size=self.batch_size,
          capacity=25000,
          enqueue_many=True,
          min_after_dequeue=0,
          num_threads=16,
          seed=self.seed,
          allow_smaller_final_batch=True,
        )

      logits = self._model(x_valid_shuffle, x_bow_valid_shuffle, d_valid_shuffle, False, reuse=True, mode="valid")
      valid_shuffle_preds = tf.argmax(logits, axis=1)
      valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
      self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
      self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
      self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)
      self.cur_valid_acc = tf.to_float(
        self.valid_shuffle_acc) / tf.to_float(self.batch_size)

  def build_model(self):

    self._build_train()
    self._build_valid()
    self._build_test()
    self._build_valid_rl()



