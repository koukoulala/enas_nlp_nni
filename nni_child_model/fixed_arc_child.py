from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import random
import numpy as np
import tensorflow as tf

from commons.utils import count_model_params, get_train_ops
from commons.common_ops import create_weight
from nni_child_model.operations import batch_norm, conv_op, pool_op
from nni_child_model.operations import recur_op, attention_op, multihead_attention, _conv_opt

class GeneralChild(object):
  def __init__(self,
               doc_train,
               bow_doc_train,
               labels_train,
               datasets_train,
               doc,
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
               cur_batch_size=32,
               eval_batch_size=100,
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
               seed=None,
               *args,
               **kwargs
              ):
    """
    """
    self.batch_size = batch_size
    self.cur_batch_size = cur_batch_size
    self.eval_batch_size = eval_batch_size
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.l2_reg = l2_reg
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_rate = lr_dec_rate
    self.keep_prob = keep_prob
    self.embed_keep_prob = embed_keep_prob
    self.lstm_x_keep_prob = lstm_x_keep_prob
    self.lstm_h_keep_prob = lstm_h_keep_prob
    self.lstm_o_keep_prob = lstm_o_keep_prob
    self.attention_keep_prob = attention_keep_prob
    self.var_rec = var_rec
    self.optim_algo = optim_algo
    self.dataset = dataset
    self.name = name
    self.seed = seed
    if self.dataset == "sst":
      self.eval_batch_size = 10

    self.global_step = None
    self.valid_acc = None
    self.test_acc = None

    if self.seed is not None:
      print("set random seed for graph building: {}".format(self.seed))
      tf.set_random_seed(self.seed)

    print("Build data ops")

    with tf.device("/cpu:0"):
      # training data
      self.num_train_examples = int(np.shape(doc["train"])[0])

      self.num_train_batches = (self.num_train_examples + self.batch_size - 1) // self.batch_size

      # doc_train,labels_train,datasets_train are placeholders
      train_dataset = tf.data.Dataset.from_tensor_slices((doc_train,
                                                          bow_doc_train, labels_train, datasets_train))

      train_dataset = train_dataset.shuffle(buffer_size=50000)
      train_dataset = train_dataset.repeat()  # repeat indefinitely
      train_dataset = train_dataset.batch(self.batch_size)
      self.train_batch_iterator = train_dataset.make_initializable_iterator()
      x_train, x_bow_train, y_train, d_train = self.train_batch_iterator.get_next()

      self.lr_dec_every = lr_dec_every * self.num_train_batches

      self.x_train = x_train
      self.y_train = y_train
      self.d_train = d_train
      self.x_bow_train = x_bow_train

      # valid data
      self.x_valid, self.x_bow_valid, self.y_valid, self.d_valid = None, None, None, None
      if doc["valid"] is not None:
        doc["valid_original"] = np.copy(doc["valid"])
        doc["valid_bow_ids_original"] = np.copy(doc["valid_bow_ids"])
        labels["valid_original"] = np.copy(labels["valid"])
        datasets["valid_original"] = np.copy(datasets["valid"])
        self.num_valid_examples = np.shape(doc["valid"])[0]
        self.num_valid_batches = (
                (self.num_valid_examples + self.eval_batch_size - 1)
                // self.eval_batch_size)

        self.doc_valid = tf.placeholder(doc["valid"].dtype,
                                        [None, doc["valid"].shape[1]], name="doc_valid")
        self.bow_doc_valid = tf.placeholder(doc["valid_bow_ids"].dtype,
                                            [None, doc["valid_bow_ids"].shape[1]], name="bow_doc_valid")
        self.labels_valid = tf.placeholder(labels["valid"].dtype,
                                           [None], name="labels_valid")
        self.datasets_valid = tf.placeholder(datasets["valid"].dtype,
                                             [None], name="datasets_valid")

        valid_dataset = tf.data.Dataset.from_tensor_slices((self.doc_valid,
                                                            self.bow_doc_valid, self.labels_valid, self.datasets_valid))
        valid_dataset = valid_dataset.batch(self.eval_batch_size)
        self.valid_batch_iterator = valid_dataset.make_initializable_iterator()
        self.x_valid, self.x_bow_valid, self.y_valid, self.d_valid = self.valid_batch_iterator.get_next()

      # test data
      self.num_test_examples = np.shape(doc["test"])[0]

      self.num_test_batches = (
              (self.num_test_examples + self.eval_batch_size - 1)
              // self.eval_batch_size)
      # change to dataset api

      self.doc_test = tf.placeholder(doc["test"].dtype,
                                     [None, doc["test"].shape[1]], name="doc_test")
      self.bow_doc_test = tf.placeholder(doc["test_bow_ids"].dtype,
                                         [None, doc["test_bow_ids"].shape[1]], name="bow_doc_test")
      self.labels_test = tf.placeholder(labels["test"].dtype,
                                        [None], name="labels_test")
      self.datasets_test = tf.placeholder(datasets["test"].dtype,
                                          [None], name="datasets_test")

      test_dataset = tf.data.Dataset.from_tensor_slices((self.doc_test,
                                                         self.bow_doc_test, self.labels_test, self.datasets_test))
      test_dataset = test_dataset.batch(self.eval_batch_size)
      self.test_batch_iterator = test_dataset.make_initializable_iterator()
      self.x_test, self.x_bow_test, self.y_test, self.d_test = self.test_batch_iterator.get_next()

    # cache doc and labels, as well as datasets
    self.doc = doc
    self.labels = labels
    self.datasets = datasets


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

    fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
    self.sample_arc = fixed_arc

  def _factorized_reduction(self, x, out_filters, stride, is_training):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      with tf.variable_scope("path_conv"):
        x = _conv_opt(x, 1, out_filters)
        x = batch_norm(x, is_training)
        return x

    actual_data_format = "channels_first"  # only support NCHW
    #stride_spec = self._get_strides(stride)
    # Skip path 1
    path1 = tf.layers.max_pooling1d(
      x, 1, stride, "VALID", data_format=actual_data_format)
    #path1 = tf.nn.max_pool(x, [1, 1, 1], stride_spec, "VALID", data_format="NCHW")
    #print("after max_pool:", path1.shape)

    with tf.variable_scope("path1_conv"):
      path1 = _conv_opt(path1, 1, out_filters // 2)

    print("after conv:", path1.shape)

    # Skip path 2
    # First pad with 0"s on the right and bottom, then shift the filter to
    # include those 0"s that were added.

    pad_arr = [[0, 0], [0, 0], [0, 1]]
    path2 = tf.pad(x, pad_arr)[:, :, 1:]
    inp_c = path2.get_shape()[1].value
    if inp_c > 1:
        concat_axis = 1
    else:
        concat_axis = 2

    path2 = tf.layers.max_pooling1d(
      path2, 1, stride, "VALID", data_format=actual_data_format)
    #path2 = tf.nn.max_pool(path2, [1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
    with tf.variable_scope("path2_conv"):
      path2 = _conv_opt(path2, 1, out_filters // 2)

    # Concat and apply BN
    final_path = tf.concat(values=[path1, path2], axis=concat_axis)
    final_path = batch_norm(final_path, is_training)

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

  def _to_sliding_window(self, doc, batch_size, size, step):
    result_doc, sliding_windows = [], []
    self.cur_batch_size = 0
    max_len = doc.get_shape()[1]
    for i in range(0, batch_size):
      start, count = 0, 0
      while start + size <= max_len:
        image = tf.slice(doc, [i, start], [1, size])
        image = tf.reshape(image, [size])
        result_doc.append(image)
        start += step
        count += 1
        self.cur_batch_size += 1
      sliding_windows.append(count)
    result_doc = tf.convert_to_tensor(result_doc)
    return result_doc, sliding_windows

  def _from_sliding_window(self, x, batch_size, sliding_windows):
    result_doc, doc = [], []
    index = 0
    print("cur_batch_size: {0}".format(self.cur_batch_size))
    print("sliding_windows: {0}".format(sliding_windows))
    for i in range(0, self.cur_batch_size):
      doc.append(x[i])
      if len(doc) == sliding_windows[index]:
        image = tf.add_n(doc)
        result_doc.append(image)
        doc = []
        index += 1
    result_doc = tf.convert_to_tensor(result_doc)
    print("result_doc: {0}".format(result_doc))
    return result_doc

  def _linear_combine(self, final_layers):
    with tf.variable_scope("linear_combine"):
      w = create_weight("w", [len(final_layers), 1, 1, 1])
      w = tf.nn.softmax(w, axis=0)
      final_layer_tensor = tf.convert_to_tensor(final_layers)
      print("final_layer_tensor: {0}".format(final_layer_tensor))
      x = tf.multiply(final_layer_tensor, w)
      x = tf.reduce_sum(x, axis=0)
      print("final_layer_tensor: {0}".format(x))
    return x

  def _model(self, doc, bow_doc, datasets, is_training,
             reuse=False, mode="train"):

    with tf.variable_scope(self.name, reuse=reuse):
      layers = []
      final_flags = []

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
        print("doc: {0}".format(doc))
        print("bow_doc: {0}".format(bow_doc))

        if is_training or mode =="valid":
          batch_size = self.batch_size
        else:
          batch_size = self.eval_batch_size

        if self.sliding_window:
          doc, sliding_windows = self._to_sliding_window(doc, batch_size, size=64, step=32)
          bow_doc, _ = self._to_sliding_window(bow_doc, batch_size, size=64, step=32)
          print("doc after sliding window: {0}".format(doc))

        if is_training:
          embedding = tf.nn.dropout(embedding, keep_prob=self.embed_keep_prob)

        doc = tf.nn.embedding_lookup(embedding, doc, max_norm=None)
        field_embedding = tf.nn.embedding_lookup(field_embedding, bow_doc, max_norm=None)
        if self.input_field_embedding:
          doc = tf.add_n([doc, field_embedding])
        doc = tf.transpose(doc, [0, 2, 1])

        print("doc_shape", doc.shape)
        inp_c = doc.shape[1]
        inp_w = doc.shape[2]
        #doc = tf.reshape(doc, [-1, inp_c, 1, inp_w])
        doc = tf.reshape(doc, [-1, inp_c, inp_w])
        field_embedding = tf.transpose(field_embedding, [0, 2, 1])
        #field_embedding = tf.reshape(field_embedding, [-1, inp_c, 1, inp_w])
        field_embedding = tf.reshape(field_embedding, [-1, inp_c, inp_w])

        print("after: doc, field_embedding", doc.shape, field_embedding.shape)

      x = doc
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
      #pos_embedding = tf.expand_dims(pos_embedding, axis=2)
      print("pos embedding: {0}".format(pos_embedding))
      if self.input_positional_encoding:
        x += pos_embedding

      out_filters = self.out_filters
      with tf.variable_scope("init_conv"):  # adjust out_filter dimension
        #print("init_x", x.shape)
        x = _conv_opt(x, 1, self.out_filters)
        x = batch_norm(x, is_training)

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


      start_idx = 0
      print("xxxxx", x)

      for layer_id in range(self.num_layers):
          with tf.variable_scope("layer_{0}".format(layer_id)):
              print("layers", layers)
              print("layer_id, x", layer_id, x)

              x = self._fixed_layer(x, pos_embedding, field_embedding, layer_id, layers,
                                        final_flags, start_idx, 0, out_filters, is_training)

              layers.append(x)
              if self.fixed_arc is not None:
                  final_flags.append(1)

              print("sample_arc: {0}".format(self.sample_arc[start_idx]))
              if layer_id in self.pool_layers:
                  layers, out_filters = add_fixed_pooling_layer(
                      layer_id, layers, out_filters, is_training, pos_embedding, field_embedding)

          start_idx += 1 + layer_id
          if self.multi_path:
              start_idx += 1
          print(layers[-1])

      print("all_layers:", layers)
      final_layers = []
      final_layers_idx = []
      for i in range(0, len(layers)):
          if self.all_layer_output:
              if self.num_last_layer_output == 0:
                  final_layers.append(layers[i])
                  final_layers_idx.append(i)
              elif i >= max((len(layers) - self.num_last_layer_output), 0):
                  final_layers.append(layers[i])
                  final_layers_idx.append(i)
          elif self.fixed_arc is not None and final_flags[i] == 1:
              final_layers.append(layers[i])
              final_layers_idx.append(i)
          elif self.fixed_arc is None:
              final_layers.append(final_flags[i] * layers[i])

      if self.fixed_arc is not None:
        print("final_layers: {0}".format(' '.join([str(idx) for idx in final_layers_idx])))

      if self.fixed_arc is not None and self.output_linear_combine:
        x = self._linear_combine(final_layers)
      else:
        x = tf.add_n(final_layers)

      if self.sliding_window:
        x = self._from_sliding_window(x, batch_size, sliding_windows)

      class_num = self.class_num
      with tf.variable_scope("fc"):
          if not self.is_output_attention:
              x = tf.reduce_mean(x, 2)
          else:
              batch_size = x.get_shape()[0].value
              inp_d = x.get_shape()[1].value
              inp_l = x.get_shape()[2].value

              final_attention_query = create_weight("query", shape=[1, inp_d], trainable=True,
                                                    initializer=tf.truncated_normal_initializer,
                                                    regularizer=regularizer)
              if is_training or mode == "valid":
                  batch_size = self.batch_size
              else:
                  batch_size = self.eval_batch_size
              final_attention_query = tf.tile(final_attention_query, [batch_size, 1])
              print("final_attention_query: {0}".format(final_attention_query))

              # put channel to the last dim
              x = tf.transpose(x, [0, 2, 1])
              x = tf.reshape(x, [-1, inp_l, inp_d])
              print("x: {0}".format(x))
              x = multihead_attention(queries=final_attention_query,
                                      keys=x,
                                      pos_embedding=pos_embedding,
                                      field_embedding=field_embedding,
                                      num_units=inp_d,
                                      num_heads=8,
                                      dropout_rate=0,
                                      is_training=is_training,
                                      causality=False,
                                      positional_encoding=self.positional_encoding)
              print("x: {0}".format(x))
              x = tf.reshape(x, [-1, 1, inp_d])
              x = tf.reduce_sum(x, axis=1)
              print("x: {0}".format(x))
          if is_training:
              x = tf.nn.dropout(x, self.keep_prob)
          x = tf.layers.dense(x, units=class_num)

    return x


  def _fixed_layer(self, inputs, pos_embedding, field_embedding, layer_id, prev_layers,
               final_flags, start_idx, pre_idx, out_filters, is_training):
      """
      Args:
        layer_id: current layer
        prev_layers: cache of previous layers. for skip connections
        start_idx: where to start looking at. technically, we can infer this
          from layer_id, but why bother...
        is_training: for batch_norm
      """
      if len(prev_layers) > 0:
          inputs = prev_layers[-1]

      if len(prev_layers) > 0:
          if self.multi_path:
              pre_layer_id = self.sample_arc[start_idx]
              start_idx += 1
              num_pre_layers = len(prev_layers)
              if num_pre_layers > 5:
                  num_pre_layers = 5
              matched = False
              for i in range(0, num_pre_layers):
                  if pre_layer_id == i:
                      layer_idx = len(prev_layers) - 1 - i
                      final_flags[layer_idx] = 0
                      matched = True
                      inputs = prev_layers[layer_idx]
              if not matched:
                  final_flags[-1] = 0
                  inputs = prev_layers[-1]
          else:
              final_flags[-1] = 0

      size = [1, 3, 5, 7]
      separables = [False, False, False, False]
      actual_data_format = "channels_first"  # NCHW

      out = inputs
      count = self.sample_arc[start_idx]
      if count in [0, 1, 2, 3]:
          filter_size = size[count]
          separable = separables[count]
          with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
              out = tf.nn.relu(out)
              out = conv_op(out, filter_size, is_training, out_filters, out_filters)
              out = batch_norm(out, is_training)
      elif count == 4:
          with tf.variable_scope("average_pool"):
              out = pool_op(out, is_training, out_filters, out_filters, "avg")
      elif count == 5:
          with tf.variable_scope("max_pool"):
              out = pool_op(out, is_training, out_filters, out_filters, "max")
      elif count == 7:
          with tf.variable_scope("out_attention"):
              out = attention_op(out, pos_embedding, field_embedding, is_training, out_filters, out_filters, start_idx=0,
                                 positional_encoding=self.positional_encoding,
                                 attention_keep_prob=self.attention_keep_prob,
                                 do_field_embedding=self.field_embedding)
              out = batch_norm(out, is_training)
      elif count == 6:
          with tf.variable_scope("rnn"):
              out = recur_op(out, is_training, out_filters, out_filters, start_idx=0,
             lstm_x_keep_prob=self.lstm_x_keep_prob, lstm_h_keep_prob=self.lstm_h_keep_prob, lstm_o_keep_prob=self.lstm_o_keep_prob, var_rec=self.var_rec)
      else:
          raise ValueError("Unknown operation number '{0}'".format(count))

      if layer_id > 0:
          skip_start = start_idx + 1
          skip = self.sample_arc[skip_start: skip_start + layer_id]
          total_skip_channels = np.sum(skip) + 1

          res_layers = []
          for i in range(layer_id):
              if skip[i] == 1:
                  res_layers.append(prev_layers[i])
                  final_flags[i] = 0
          prev = res_layers + [out]

          if not self.skip_concat:
              out = tf.add_n(prev)
          else:
              prev = tf.concat(prev, axis=1)
              out = prev
              print(out, out_filters)
              with tf.variable_scope("skip"):
                  out = tf.nn.relu(out)
                  out = conv_op(out, 1, is_training, out_filters, out_filters)
          out = batch_norm(out, is_training)

      return out

  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False, global_step=None):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    if global_step is None:
      assert self.global_step is not None
      global_step = sess.run(self.global_step)
    print("Eval at {}".format(global_step))

    if eval_set == "valid":
      assert self.x_valid is not None
      assert self.valid_acc is not None
      num_examples = self.num_valid_examples
      num_batches = self.num_valid_batches
      acc_op = self.valid_acc
      sess.run(self.valid_batch_iterator.initializer, feed_dict={
          self.doc_valid: self.doc["valid"],
          self.bow_doc_valid: self.doc["valid_bow_ids"],
          self.labels_valid: self.labels["valid"],
          self.datasets_valid: self.datasets["valid"]
      })
    elif eval_set == "test":
      assert self.test_acc is not None
      num_examples = self.num_test_examples
      num_batches = self.num_test_batches
      acc_op = self.test_acc
      sess.run(self.test_batch_iterator.initializer, feed_dict={
          self.doc_test: self.doc["test"],
          self.bow_doc_test: self.doc["test_bow_ids"],
          self.labels_test: self.labels["test"],
          self.datasets_test: self.datasets["test"]
      })
    else:
      raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

    total_acc = 0
    total_exp = 0
    batch_id = 0
    while True:
      try:
        acc = sess.run(acc_op, feed_dict=feed_dict)
        total_acc += acc
        if (batch_id + 1) == num_batches and num_examples % self.eval_batch_size != 0:
          total_exp += num_examples % self.eval_batch_size
        else:
          total_exp += self.eval_batch_size
        if verbose:
          sys.stdout.write("\r{:<5d}/{:>5d}".format(total_acc, total_exp))
        batch_id += 1
      except tf.errors.OutOfRangeError:
        break
    if verbose:
      print("")
    if total_exp > 0:
      final_acc = float(total_acc) / total_exp
      print("{}_accuracy: {:<6.4f}".format(eval_set, final_acc))
    else:
      final_acc = 0
      print("Error in calculating final_acc")

    return final_acc


  # override
  def _build_train(self):
      self._build_train_cat()

  def _build_train_cat(self):
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
          # shuffled valid data: for choosing validation model
          valid_original = self.doc["valid_original"]
          valid_bow_original = self.doc["valid_bow_ids_original"]
          x_valid_shuffle, x_bow_valid_shuffle, y_valid_shuffle, d_valid_shuffle = tf.train.shuffle_batch(
              [valid_original, valid_bow_original, self.labels["valid_original"], self.datasets["valid_original"]],
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



