from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from commons.utils import get_train_ops
from commons.common_ops import stack_lstm


class GeneralController():
    def __init__(self,
                 search_for="both",
                 num_layers=4,
                 num_branches=6,
                 out_filters=48,
                 lstm_size=32,
                 lstm_num_layers=2,
                 lstm_keep_prob=1.0,
                 tanh_constant=None,
                 temperature=None,
                 lr_init=1e-3,
                 lr_dec_start=0,
                 lr_dec_every=100,
                 lr_dec_rate=0.9,
                 l2_reg=0,
                 entropy_weight=None,
                 clip_mode=None,
                 grad_bound=None,
                 bl_dec=0.999,
                 optim_algo="adam",
                 skip_target=0.8,
                 skip_weight=0.5,
                 pre_idxs=[],
                 multi_path=True,
                 name="controller",
                 batch_size=20,
                 *args,
                 **kwargs):

        print("-" * 80)
        print("Building Controller")

        self.multi_path = multi_path
        self.search_for = search_for
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_keep_prob = lstm_keep_prob
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.bl_dec = bl_dec

        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self.optim_algo = optim_algo
        self.pre_idxs = pre_idxs
        self.name = name
        self.batch_size = batch_size

        print('begin to create params...\n')
        self._create_params()
        self._build_sampler(pre_idxs)

    def _create_params(self):
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name, initializer=initializer):
            with tf.variable_scope("lstm"):
                self.w_lstm = []
                for layer_id in range(self.lstm_num_layers):
                    with tf.variable_scope("layer_{}".format(layer_id)):
                        w = tf.get_variable(
                            "w", [2 * self.lstm_size, 4 * self.lstm_size])
                        self.w_lstm.append(w)
                        print('create get variable done...\n')

            self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])
            with tf.variable_scope("emb"):
                self.w_emb = tf.get_variable(
                    "w", [self.num_branches, self.lstm_size])
                self.layer_emb = tf.get_variable(
                    "w_layer", [self.num_layers, self.lstm_size])

            with tf.variable_scope("softmax"):
                self.w_soft = tf.get_variable(
                    "w", [self.lstm_size, self.num_branches])

            with tf.variable_scope("layer_softmax"):
                self.w_layer = tf.get_variable(
                    "w", [self.lstm_size, 5])

            with tf.variable_scope("attention"):
                self.w_attn_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
                self.w_attn_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
                self.v_attn = tf.get_variable("v", [self.lstm_size, 1])

    def _build_sampler(self, pre_idxs=[]):
        """Build the sampler ops and the log_prob ops."""

        print("-" * 80)
        print("Build controller sampler")

        for _ in range(self.batch_size):
            anchors = []
            anchors_w_1 = []

            arc_seq = []
            entropys = []
            log_probs = []
            skip_count = []
            skip_penaltys = []

            prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                      range(self.lstm_num_layers)]
            prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                      range(self.lstm_num_layers)]
            inputs = self.g_emb
            skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
                                       dtype=tf.float32)
            idx = 0
            for layer_id in range(self.num_layers):
                # choose a previous layer as input
                if self.multi_path == True:
                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    pre_num_layers = layer_id + 1
                    if pre_num_layers > 5:
                        pre_num_layers = 5
                    left_num_layers = 5 - pre_num_layers
                    mask1 = tf.fill([self.lstm_size, pre_num_layers], True)
                    mask2 = tf.fill([self.lstm_size, left_num_layers], False)
                    mask = tf.concat([mask1, mask2], 1)
                    w_layer = tf.boolean_mask(self.w_layer, mask)
                    w_layer = tf.reshape(w_layer, [self.lstm_size, pre_num_layers])

                    logit = tf.matmul(next_h[-1], w_layer)

                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)
                    if self.search_for == "macro" or self.search_for == "branch":
                        if idx < len(pre_idxs):
                            input_layer_id = pre_idxs[idx]
                            idx += 1
                        else:
                            input_layer_id = tf.multinomial(logit, 1)
                            input_layer_id = tf.to_int32(input_layer_id)
                            input_layer_id = tf.reshape(input_layer_id, [1])
                    else:
                        raise ValueError("Unknown search_for {}".format(self.search_for))

                    arc_seq.append(input_layer_id)
                    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logit, labels=input_layer_id)
                    log_probs.append(log_prob)
                    entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
                    entropys.append(entropy)
                    inputs = tf.nn.embedding_lookup(self.layer_emb, input_layer_id)

                # choose branch id
                next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h
                logit = tf.matmul(next_h[-1], self.w_soft)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * tf.tanh(logit)
                if self.search_for == "macro" or self.search_for == "branch":
                    if idx < len(pre_idxs):
                        branch_id = pre_idxs[idx]
                        idx += 1
                    else:
                        branch_id = tf.multinomial(logit, 1)
                    branch_id = tf.to_int32(branch_id)
                    branch_id = tf.reshape(branch_id, [1])
                elif self.search_for == "connection":
                    branch_id = tf.constant([0], dtype=tf.int32)
                else:
                    raise ValueError("Unknown search_for {}".format(self.search_for))
                arc_seq.append(branch_id)
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logit, labels=branch_id)
                log_probs.append(log_prob)
                entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
                entropys.append(entropy)
                inputs = tf.nn.embedding_lookup(self.w_emb, branch_id)

                # set skip connections
                next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h

                if layer_id > 0:
                    query = tf.concat(anchors_w_1, axis=0)
                    query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
                    query = tf.matmul(query, self.v_attn)
                    logit = tf.concat([-query, query], axis=1)
                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)

                    skip = tf.multinomial(logit, 1)
                    skip = tf.to_int32(skip)
                    skip = tf.reshape(skip, [layer_id])
                    arc_seq.append(skip)

                    skip_prob = tf.sigmoid(logit)
                    kl = skip_prob * tf.log(skip_prob / skip_targets)
                    kl = tf.reduce_sum(kl)
                    skip_penaltys.append(kl)

                    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logit, labels=skip)
                    log_probs.append(tf.reduce_sum(log_prob, keep_dims=True))

                    entropy = tf.stop_gradient(
                        tf.reduce_sum(log_prob * tf.exp(-log_prob), keep_dims=True))
                    entropys.append(entropy)

                    skip = tf.to_float(skip)
                    skip = tf.reshape(skip, [1, layer_id])
                    skip_count.append(tf.reduce_sum(skip))
                    inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
                    inputs /= (1.0 + tf.reduce_sum(skip))
                else:
                    inputs = self.g_emb

                anchors.append(next_h[-1])
                anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))

            arc_seq = tf.concat(arc_seq, axis=0)
            self.sample_arc = tf.reshape(arc_seq, [-1])

            entropys = tf.stack(entropys)
            self.sample_entropy = tf.reduce_sum(entropys)

            log_probs = tf.stack(log_probs)
            self.sample_log_prob = tf.reduce_sum(log_probs)

            skip_count = tf.stack(skip_count)
            self.skip_count = tf.reduce_sum(skip_count)

            skip_penaltys = tf.stack(skip_penaltys)
            self.skip_penaltys = tf.reduce_mean(skip_penaltys)

    def build_trainer(self):
        self.valid_acc = tf.placeholder(dtype=tf.float32, shape=[])
        reward = self.valid_acc

        normalize = tf.to_float(self.num_layers * (self.num_layers - 1) / 2)
        self.skip_rate = tf.to_float(self.skip_count) / normalize

        if self.entropy_weight is not None:
            reward += self.entropy_weight * self.sample_entropy

        self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
        self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        baseline_update = tf.assign_sub(
            self.baseline, (1 - self.bl_dec) * (self.baseline - reward))

        with tf.control_dependencies([baseline_update]):
            reward = tf.identity(reward)

        self.loss = self.sample_log_prob * (reward - self.baseline)
        if self.skip_weight is not None:
            self.loss += self.skip_weight * self.skip_penaltys

        self.train_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="train_step")
        tf_variables = [var
                        for var in tf.trainable_variables() if var.name.startswith(self.name)]
        print("-" * 80)
        for var in tf_variables:
            print(var)

        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
            self.loss,
            tf_variables,
            self.train_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            optim_algo=self.optim_algo)

