import os
import sys

import numpy as np
import tensorflow as tf


class Model(object):
  def __init__(self,
               doc_train,
               bow_doc_train,
               labels_train,
               datasets_train,
               doc,
               labels,
               datasets,
               batch_size=32,
               cur_batch_size=32,
               eval_batch_size=100,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.1,
               keep_prob=1.0,
               embed_keep_prob=1.0,
               lstm_x_keep_prob=1.0,
               lstm_h_keep_prob=1.0,
               lstm_o_keep_prob=1.0,
               attention_keep_prob=1.0,
               var_rec=False,
               optim_algo=None,
               data_format="NHWC",
               dataset="sst",
               name="generic_model",
               seed=None
              ):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    print("-" * 80)
    print("Build model {}".format(name))

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
    self.data_format = data_format
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

      self.num_train_batches = (
        self.num_train_examples + self.batch_size - 1) // self.batch_size

      # doc_train,labels_train,datasets_train are placeholders
      train_dataset = tf.data.Dataset.from_tensor_slices((doc_train,
                                           bow_doc_train, labels_train, datasets_train))

      train_dataset = train_dataset.shuffle(buffer_size=50000)
      train_dataset = train_dataset.repeat() # repeat indefinitely
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

  def _model(self, doc, is_training, reuse=None):
    raise NotImplementedError("Abstract method")

  def _build_train(self):
    raise NotImplementedError("Abstract method")

  def _build_valid(self):
    raise NotImplementedError("Abstract method")

  def _build_test(self):
    raise NotImplementedError("Abstract method")
