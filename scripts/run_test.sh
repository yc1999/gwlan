#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python main.py \
    --strategy ddp \
    --check_val_every_n_epoch 1 \
    --train_dir ./dataset/en2de/train/train.en2de.de.out.bi_context \
    --valid_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --test_dir ./dataset/wmt14_test_data/bi_context_raw_data/test.en2de.de.out.bi_context \
    --model_name_or_path wpm \
    --train_batch_size 50 \
    --eval_batch_size 32 \
    --max_seq_len 128 \
    --max_epochs 250000 \
    --gpus 1 \
    --accumulate_grad_batches 5 \
    --vocab_size 50005 \
    --d_model 512 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --scheduler linear \
    --do_test \
    --setting debug \
    --track_grad_norm 2 \
    --gradient_clip_val 1.0 \
    --ckpt_path "/home/yc21/project/gwlan/save/wpm/seed: 42 - max_epochs: -1 - max_steps: 250000 - warmup_ratio: 0.1 - warmup_steps: -1 - train_dataset_ratio: 1.0 - train_bacth_size: 50 - gpus: 4 - accumulate_grad_batches: 5 - lr: 5e-05 -scheduler: linear - dropout: 0.1 - gradient_clip_val: 1.0/epoch=18--valid_acc=0.4163.ckpt" 
    # --do_train \
    # --test_dir ./dataset/wmt14_test_data/bi_context_raw_data/test.en2de.de.out.bi_context \