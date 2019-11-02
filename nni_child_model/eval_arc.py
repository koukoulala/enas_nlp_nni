from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time
import tensorflow as tf
import logging
import numpy as np

from nni_child_model.data.data_utils import read_data_sst
from nni_child_model.fixed_arc_child import GeneralChild
from commons.flags import FLAGS


def build_logger(log_name):
  logger = logging.getLogger(log_name)
  logger.setLevel(logging.DEBUG)
  fh = logging.FileHandler(log_name + '.log')
  fh.setLevel(logging.DEBUG)
  logger.addHandler(fh)
  return logger


logger = build_logger("nni_child_enas_nlp")


def get_model(images_train, bow_images_train, labels_train, datasets_train,
              images, labels, datasets, embedding, num_layers):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """

  print("num layers: {0}".format(num_layers))
  assert FLAGS.search_for is not None, "Please specify --search_for"

  ChildClass = GeneralChild

  child_model = ChildClass(
    images_train,
    bow_images_train,
    labels_train,
    datasets_train,
    images,
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
    sliding_window=FLAGS.sliding_window
  )

  return child_model


def get_ops(child_model):

  child_model.build_model()
  child_ops = {
    "global_step": child_model.global_step,
    "loss": child_model.loss,
    "train_op": child_model.train_op,
    "lr": child_model.lr,
    "grad_norm": child_model.grad_norm,
    "train_acc": child_model.train_acc,
    "valid_acc": child_model.valid_acc,
    "optimizer": child_model.optimizer,
    "num_train_batches": child_model.num_train_batches,
    "train_batch_iterator": child_model.train_batch_iterator,
    "valid_lengths": child_model.valid_lengths
  }

  if FLAGS.child_lr_decay_scheme == "auto":
    child_ops["num_valid_batches"] = child_model.num_valid_batches
    child_ops["step_value"] = child_model.step_value
    child_ops["assign_global_step"] = child_model.assign_global_step

  ops = {
    "child": child_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }

  return ops

# please see: https://github.com/tensorflow/tensorflow/issues/8425
def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    session = session._sess
  return session


def train(doc, labels, datasets, embedding):

  g = tf.Graph()
  with g.as_default():
    with tf.device("/cpu:0"):
      print(doc["train"])
      doc_train = tf.placeholder(doc["train"].dtype,
                     [None, doc["train"].shape[1]], name="doc_train")
      bow_doc_train = tf.placeholder(doc["train_bow_ids"].dtype,
                     [None, doc["train_bow_ids"].shape[1]], name="bow_doc_train")
      labels_train = tf.placeholder(labels["train"].dtype,
                     [None], name="labels_train")
      datasets_train = tf.placeholder(datasets["train"].dtype,
                     [None], name="datasets_train")

      print("child_num_branches {0}".format(str(FLAGS.child_num_branches)))

    child_model = get_model(doc_train, bow_doc_train,labels_train, datasets_train, doc, labels, datasets, embedding,
                            FLAGS.child_num_layers)

    print("finish get ori_ops")
    ops = get_ops(child_model)
    print("finish get ops")

    child_ops = ops["child"]

    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)

    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      FLAGS.output_dir, save_steps=10000, saver=saver)

    hooks = [checkpoint_saver_hook]

    if FLAGS.child_lr_decay_scheme == "auto":
      child_lr = FLAGS.child_lr
      child_warmup_lr = FLAGS.child_lr / 10
      child_warmup_epochs = 5
      slide_window_size = 7
      latest_eval_accs = []
      need_decay_lr = False
      enable_decay = False
      decay_factors = [0.2, 0.2, 0.2, 0.2]
      decays_count = 0
      best_valid_ckpt_dir = FLAGS.output_dir + '_best_ckpt'
      best_valid_ckpt_file = "valid_best.ckpt"
      best_ckpt_saver = tf.train.Saver(max_to_keep=1)
      best_acc = 0.0

      if not os.path.isdir(best_valid_ckpt_dir):
        print("Path {} does not exist. Creating.".format(best_valid_ckpt_dir))
        os.makedirs(best_valid_ckpt_dir)
      else:
        print("Path {} exists. Remove and remake.".format(best_valid_ckpt_dir))
        shutil.rmtree(best_valid_ckpt_dir, ignore_errors=True)
        os.makedirs(best_valid_ckpt_dir)

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
        start_time = time.time()
        sess.run(child_ops["train_batch_iterator"].initializer, feed_dict={
                 doc_train: doc["train"],
                 bow_doc_train: doc["train_bow_ids"],
                 labels_train: labels["train"],
                 datasets_train: datasets["train"]})
        print("finish initializing training dataset iterator")

        epoch = 0
        threshold = -1
        valid_acc = 0
        while True:
          run_ops = [
            child_ops["loss"],
            child_ops["lr"],
            child_ops["grad_norm"],
            child_ops["train_acc"],
            child_ops["train_op"],
            child_ops["valid_lengths"]
          ]

          if FLAGS.child_lr_decay_scheme == "auto":
            if enable_decay:
              if need_decay_lr and decays_count >= len(decay_factors):
                break
              elif need_decay_lr:
                child_lr *= decay_factors[decays_count]
                decays_count += 1
                need_decay_lr = False
                global_step = sess.run(child_ops["global_step"])
                # continue training from the best checkpoint
                saver.restore(get_session(sess), os.path.join(best_valid_ckpt_dir, best_valid_ckpt_file))
                print("Decay learning rate, current lr: {}, continue from {}".format(child_lr, sess.run(child_ops["global_step"])))
                # print("Expected step value: {}, restored global step: {}".format(
                #     global_step, sess.run(child_ops["global_step"])))
                sess.run(child_ops["assign_global_step"], feed_dict={child_ops["step_value"]: global_step})
                # print("Assigned global step value: {}".format(sess.run(child_ops["global_step"])))
                best_acc = 0.0 # reset best_acc
            else:
              child_lr = child_warmup_lr
              if epoch >= child_warmup_epochs:
                enable_decay = True
                child_lr = FLAGS.child_lr
                print("Enable to decay, best valid acc: {}".format(best_acc))
                best_acc = 0.0 # reset best_acc
                continue
            loss, lr, gn, tr_acc, _, valid_lengths = sess.run(run_ops, feed_dict={child_ops["lr"]: child_lr})
            if FLAGS.is_debug:
              print(valid_lengths)
          else:
            loss, lr, gn, tr_acc, _, valid_lengths = sess.run(run_ops)
            if FLAGS.is_debug:
              print(valid_lengths)

          global_step = sess.run(child_ops["global_step"])
          #writer = tf.summary.FileWriter("child_graph")
          #writer.add_graph(sess.graph)

          actual_step = global_step

          epoch = actual_step // ops["num_train_batches"]
          curr_time = time.time()
          if global_step % FLAGS.log_every == 0:
            log_string = ""
            log_string += "epoch={:<6d}".format(epoch)
            log_string += "ch_step={:<6d}".format(global_step)
            log_string += " loss={:<8.6f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " |g|={:<8.4f}".format(gn)
            log_string += " tr_acc={:<3d}/{:>3d}".format(
              tr_acc, FLAGS.batch_size)
            log_string += " mins={:<10.2f}".format(
                float(curr_time - start_time) / 60)
            print(log_string)

          if actual_step % ops["eval_every"] == 0:

            print("Epoch {}: Eval".format(epoch))
            if FLAGS.child_lr_decay_scheme == "auto":
              acc = ops["eval_func"](sess, "valid")
              if enable_decay:
                if len(latest_eval_accs) < slide_window_size:
                  latest_eval_accs.append(acc)
                else:
                  old_avg_acc = np.mean(latest_eval_accs)
                  del latest_eval_accs[0]
                  latest_eval_accs.append(acc)
                  cur_avg_acc = np.mean(latest_eval_accs)
                  avg_acc_diff = cur_avg_acc - old_avg_acc
                  #if avg_acc_diff < 0 and abs(avg_acc_diff) / old_avg_acc > acc_drop_ratio_threshold:
                  if avg_acc_diff < 0: # decay when average acc drop
                    need_decay_lr = True

              if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(best_valid_ckpt_dir, best_valid_ckpt_file)
                best_ckpt_saver.save(get_session(sess), save_path)
                print("Found better model at: {}".format(global_step))
            else:
              acc = ops["eval_func"](sess, "valid")

            ops["eval_func"](sess, "test") # always evaluate test data

          if epoch >= FLAGS.num_epochs:
            break

def main(_):
  print("-" * 80)

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

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")

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

  return train(doc, labels, datasets, embedding)

if __name__ == "__main__":
  tf.app.run()