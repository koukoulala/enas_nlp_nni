#!/bin/bash
set -e
export PYTHONPATH="$(pwd)"

python3 ./nni_child_model/entry.py \
  --train_ratio=1.0 \
  --valid_ratio=1.0 \
  --embedding_model="none" \
  --multi_path \
  --min_count=1 \
  --is_mask \
  --all_layer_output \
  --output_linear_combine \
  --child_lr_decay_scheme="cosine" \
  --search_for="macro" \
  --reset_output_dir \
  --data_path="./data/sst_test" \
  --dataset="sst" \
  --class_num=5 \
  --pool_step=4 \
  --child_optim_algo="momentum" \
  --data_type="text" \
  --max_input_length=32 \
  --output_dir="outputs" \
  --train_data_size=45000 \
  --batch_size=16 \
  --num_epochs=150 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_use_aux_heads \
  --child_num_layers=12 \
  --child_out_filters=32 \
  --child_l2_reg=0.00002 \
  --child_num_branches=8 \
  --child_start_branches=6 \
  --child_progressive_branches=8 \
  --child_grad_bound=5.0 \
  --child_num_cell_layers=5 \
  --child_keep_prob=0.5 \
  --embed_keep_prob=0.5 \
  --lstm_x_keep_prob=0.5 \
  --lstm_h_keep_prob=1.0 \
  --lstm_o_keep_prob=0.5 \
  --attention_keep_prob=0.5 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine \
  --child_lr=0.005 \
  --child_lr_max=0.005 \
  --child_lr_min=0.0001 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_search_whole_channels \
  --controller_train_steps=20 \
  --child_mode="subgraph" \

  "$@"
