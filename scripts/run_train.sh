#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

python main.py \
    --strategy ddp \
    --val_check_interval 1.0 \
    --train_dir ./dataset/en2de/train/train.en2de.de.out.bi_context \
    --valid_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --test_dir ./dataset/wmt14_test_data/bi_context_raw_data/test.en2de.de.out.bi_context \
    --model_name_or_path wpm \
    --train_batch_size 64 \
    --max_seq_len 128 \
    --max_epochs 100 \
    --gpus 2 \
    --accumulate_grad_batches 1 \
    --vocab_size 50005 \
    --d_model 512 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --do_train \
    --do_test \
    --setting \
    run
