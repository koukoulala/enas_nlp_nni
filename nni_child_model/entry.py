from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import nni

import tensorflow as tf
import gensim.models.keyedvectors as word2vec
import logging

from nni_child_model.data.data_utils import read_data_sst
from nni_child_model.nni_child import GeneralChild
from commons.flags import FLAGS


def build_logger(log_name):
  logger = logging.getLogger(log_name)
  logger.setLevel(logging.DEBUG)
  fh = logging.FileHandler(log_name + '.log')
  fh.setLevel(logging.DEBUG)
  logger.addHandler(fh)
  return logger


logger = build_logger("nni_child_enas_nlp")

def build_trial(doc_train, bow_doc_train, labels_train, datasets_train,
              doc, labels, datasets, embedding, num_layers, ChildClass):
  '''Build child class'''
  child_model = ChildClass(
    doc_train,
    bow_doc_train,
    labels_train,
    datasets_train,
    doc,
    labels,
    datasets,
    embedding,
    num_layers=num_layers,
    num_cells=FLAGS.child_num_cells,
    num_branches=FLAGS.child_num_branches,
    fixed_arc=FLAGS.child_fixed_arc,
    out_filters_scale=FLAGS.child_out_filters_scale,
    out_filters=FLAGS.child_out_filters,
    keep_prob=FLAGS.child_keep_prob,
    embed_keep_prob=FLAGS.embed_keep_prob,
    lstm_x_keep_prob=FLAGS.lstm_x_keep_prob,
    lstm_h_keep_prob=FLAGS.lstm_h_keep_prob,
    lstm_o_keep_prob=FLAGS.lstm_o_keep_prob,
    attention_keep_prob=FLAGS.attention_keep_prob,
    var_rec=FLAGS.var_rec,
    num_epochs=FLAGS.num_epochs,
    l2_reg=FLAGS.child_l2_reg,
    dataset=FLAGS.dataset,
    class_num=FLAGS.class_num,
    pool_step=FLAGS.pool_step,
    batch_size=FLAGS.batch_size,
    clip_mode="global",
    grad_bound=FLAGS.child_grad_bound,
    lr_warmup_steps=FLAGS.child_lr_warmup_steps,
    lr_model_d=FLAGS.child_lr_model_d,
    lr_init=FLAGS.child_lr,
    lr_dec_start=FLAGS.child_lr_dec_start,
    lr_dec_every=FLAGS.child_lr_dec_every,
    lr_dec_rate=FLAGS.child_lr_dec_rate,
    lr_decay_scheme=FLAGS.child_lr_decay_scheme,
    lr_decay_epoch_multiplier=FLAGS.child_lr_decay_epoch_multiplier,
    lr_max=FLAGS.child_lr_max,
    lr_min=FLAGS.child_lr_min,
    lr_T_0=FLAGS.child_lr_T_0,
    lr_T_mul=FLAGS.child_lr_T_mul,
    optim_algo=FLAGS.child_optim_algo,
    multi_path=FLAGS.multi_path,
    positional_encoding=FLAGS.positional_encoding,
    input_positional_encoding=FLAGS.input_positional_encoding,
    is_sinusolid=FLAGS.is_sinusolid,
    embedding_model=FLAGS.embedding_model,
    skip_concat=FLAGS.skip_concat,
    all_layer_output=FLAGS.all_layer_output,
    num_last_layer_output=FLAGS.num_last_layer_output,
    output_linear_combine=FLAGS.output_linear_combine,
    is_debug=FLAGS.is_debug,
    is_mask=FLAGS.is_mask,
    is_output_attention=FLAGS.is_output_attention,
    field_embedding=FLAGS.field_embedding,
    input_field_embedding=FLAGS.input_field_embedding,
    sliding_window=FLAGS.sliding_window,
    seed=None
  )

  return child_model


def get_child_ops(child_model):
    '''Assemble child op to a dict'''
    child_ops = {
      "global_step": child_model.global_step,
      "loss": child_model.loss,
      "train_op": child_model.train_op,
      "lr": child_model.lr,
      "grad_norm": child_model.grad_norm,
      "train_acc": child_model.train_acc,
      "optimizer": child_model.optimizer,
      "num_train_batches": child_model.num_train_batches,
      "train_batch_iterator": child_model.train_batch_iterator,
      "valid_lengths": child_model.valid_lengths,
      "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
      "eval_func": child_model.eval_once,
    }
    return child_ops


class NASTrial():

  def __init__(self, doc, labels, datasets, embedding):

    self.output_dir = os.path.join(os.getenv('NNI_OUTPUT_DIR'), '../..')
    self.file_path = os.path.join(
      self.output_dir, 'trainable_variable.txt')

    self.graph = tf.Graph()
    with self.graph.as_default():
      with tf.device("/cpu:0"):
        print(doc["train"].shape, doc["train"])
        doc_train = tf.placeholder(doc["train"].dtype,
                                      [None, doc["train"].shape[1]], name="doc_train")
        bow_doc_train = tf.placeholder(doc["train_bow_ids"].dtype,
                                          [None, doc["train_bow_ids"].shape[1]], name="bow_doc_train")
        labels_train = tf.placeholder(labels["train"].dtype,
                                      [None], name="labels_train")
        datasets_train = tf.placeholder(datasets["train"].dtype,
                                        [None], name="datasets_train")

        print("child_num_branches {0}".format(str(FLAGS.child_num_branches)))


      self.child_model = build_trial(doc_train, bow_doc_train,
        labels_train, datasets_train, doc, labels, datasets, embedding,
        FLAGS.child_num_layers, GeneralChild)

      self.total_data = {}

      self.child_model.build_model()
      self.child_ops = get_child_ops(self.child_model)
      config = tf.ConfigProto(
        intra_op_parallelism_threads=0,
        inter_op_parallelism_threads=0,
        allow_soft_placement=True)

      self.sess = tf.train.SingularMonitoredSession(config=config)
      self.sess.run(self.child_ops["train_batch_iterator"].initializer, feed_dict={
          doc_train: doc["train"],
          bow_doc_train: doc["train_bow_ids"],
          labels_train: labels["train"],
          datasets_train: datasets["train"]})

    logger.debug('initlize NASTrial done.')

  def run_one_step(self):
    '''Run this model on a batch of data'''
    run_ops = [
      self.child_ops["loss"],
      self.child_ops["lr"],
      self.child_ops["grad_norm"],
      self.child_ops["train_acc"],
      self.child_ops["train_op"],
      self.child_ops["valid_lengths"],
    ]
    loss, lr, gn, tr_acc, _, valid_lengths = self.sess.run(run_ops)
    global_step = self.sess.run(self.child_ops["global_step"])
    log_string = ""
    log_string += "ch_step={:<6d}".format(global_step)
    log_string += " loss={:<8.6f}".format(loss)
    log_string += " lr={:<8.4f}".format(lr)
    log_string += " |g|={:<8.4f}".format(gn)
    log_string += " tr_acc={:<3d}/{:>3d}".format(tr_acc, FLAGS.batch_size)
    if int(global_step) % FLAGS.log_every == 0:
      logger.debug(log_string)
    return loss, global_step

  def get_csvaa(self):
      cur_valid_acc = self.sess.run(self.child_model.cur_valid_acc)
      return cur_valid_acc

  def run(self, num):
      for _ in range(num):
          """@nni.get_next_parameter(self.sess)"""
          #"""@nni.training_update(tf=tf, session=self.sess)"""
          """@nni.variable(nni.choice('train', 'validate'), name=entry)"""
          entry = 'trian'
          if entry == 'train':
              loss, _ = self.run_one_step()
              '''@nni.report_final_result(loss)'''
          elif entry == 'validate':
              valid_acc_arr = self.get_csvaa()
              '''@nni.report_final_result(valid_acc_arr)'''
          else:
              raise RuntimeError('No such entry: ' + entry)


def main(_):
    logger.debug("-" * 80)

    if not os.path.isdir(FLAGS.output_dir):
        logger.debug(
            "Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        logger.debug(
            "Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
    logger.debug("-" * 80)

    is_valid = True
    word_id_dict, word_num_dict = {}, {}

    if FLAGS.dataset == 'sst':
      cache = {}
      doc, labels, datasets, embedding = read_data_sst(word_id_dict, word_num_dict,
                                                          FLAGS.data_path, FLAGS.max_input_length,
                                                          FLAGS.embedding_model,
                                                          FLAGS.min_count, FLAGS.train_ratio, FLAGS.valid_ratio,
                                                          FLAGS.is_binary, is_valid,
                                                          cache=cache)
    else:
      print("Unknown dataset name!")

    trial = NASTrial(doc, labels, datasets, embedding)

    trial.run(400*FLAGS.num_epochs)


if __name__ == "__main__":
    tf.app.run()

