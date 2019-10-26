from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from commons.utils import DEFINE_boolean
from commons.utils import DEFINE_float
from commons.utils import DEFINE_integer
from commons.utils import DEFINE_string
flags = tf.app.flags
FLAGS = flags.FLAGS


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("search_for", "macro", "Must be [macro|micro]")
DEFINE_string("dataset", "sst", "dataset for evaluation")
DEFINE_string("embedding_model", "word2vec", "word2vec or glove")
DEFINE_string("child_lr_decay_scheme", "exponential", "Strategy to decay learning "
              "rate, must be ['cosine', 'noam', 'exponential', 'auto']")

DEFINE_integer("child_lr_decay_epoch_multiplier", 1, "")
DEFINE_integer("train_data_size", 45000, "")
DEFINE_integer("batch_size", 128, "")
DEFINE_integer("max_input_length", 50, "")
DEFINE_integer("class_num", 5, "")
DEFINE_integer("pool_step", 3, "")

DEFINE_integer("num_epochs", 310, "")
DEFINE_integer("child_lr_dec_start", 0, "")
DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_integer("child_lr_warmup_steps", 100, "")
DEFINE_integer("child_lr_model_d", 300, "")
DEFINE_integer("child_num_layers", 12, "")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 36, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 6, "")
DEFINE_integer("child_lr_T_0", None, "for lr schedule")
DEFINE_integer("child_lr_T_mul", None, "for lr schedule")
DEFINE_integer("min_count", 0, "")
DEFINE_float("train_ratio", 0.5, "")
DEFINE_float("valid_ratio", 0.1, "")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.5, "")
DEFINE_float("lstm_x_keep_prob", 0.5, "")
DEFINE_float("lstm_h_keep_prob", 0.8, "")
DEFINE_float("lstm_o_keep_prob", 0.5, "")
DEFINE_float("embed_keep_prob", 0.8, "")
DEFINE_float("attention_keep_prob", 1.0, "")
DEFINE_float("child_l2_reg", 1e-4, "")
DEFINE_float("child_lr_max", None, "for lr schedule")
DEFINE_float("child_lr_min", None, "for lr schedule")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_string("child_optim_algo", "momentum", "")
DEFINE_boolean("is_binary", False, "binary label for sst dataset")
DEFINE_boolean("positional_encoding", False, "use positional encoding on attention input")
DEFINE_boolean("input_positional_encoding", False, "use positional encoding as input")
DEFINE_boolean("is_sinusolid", False, "use sinusolid as positional encoding")
DEFINE_boolean("var_rec", False, "varational_recurrent")
DEFINE_boolean("skip_concat", False, "use concat in skip connection")
DEFINE_boolean("all_layer_output", False, "use all layer as output")
DEFINE_integer("num_last_layer_output", 0, "last n layers as output, 0 for all")
DEFINE_boolean("output_linear_combine", False, "linear combine of output layers")
DEFINE_boolean("is_debug", False, "whether print debug info")
DEFINE_boolean("is_mask", False, "whether apply mask")
DEFINE_boolean("is_output_attention", False, "apply attention before softmax output")
DEFINE_boolean("field_embedding", False, "whether use field embedding in attention")
DEFINE_boolean("input_field_embedding", False, "whether use field embedding in input")
DEFINE_boolean("sliding_window", False, "use sliding window as input")

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_op_tanh_reduce", 1.0, "")
DEFINE_float("controller_entropy_weight", 0.0001, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 1,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", True, "")
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

DEFINE_string("tuner_class_name", "", "")
DEFINE_string("tuner_class_filename", "", "")
DEFINE_string("tuner_args", "", "")
DEFINE_string("tuner_directory", "", "")
DEFINE_string("assessor_class_name", "", "")
DEFINE_string("assessor_args", "", "")
DEFINE_string("assessor_directory", "", "")
DEFINE_string("assessor_class_filename", "", "")
DEFINE_boolean("multi_phase", True, "")
DEFINE_boolean("multi_thread", True, "")
DEFINE_boolean("multi_path", True, "Search for multiple path in the architecture")
