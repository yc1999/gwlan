#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,4

python main.py \
    --train_dataset_ratio 1.0 \
    --strategy ddp \
    --val_check_interval 250 \
    --train_dir ./dataset/en2de/train/train.en2de.de.out.bi_context \
    --valid_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --test_dir ./dataset/wmt14_test_data/bi_context_raw_data/test.en2de.de.out.bi_context \
    --model_name_or_path vanilla \
    --train_batch_size 100 \
    --max_seq_len 128 \
    --max_epochs -1 \
    --max_steps 250000 \
    --gpus 2 \
    --accumulate_grad_batches 1 \
    --vocab_size 50005 \
    --d_model 512 \
    --learning_rate 5e-5 \
    --scheduler linear \
    --warmup_ratio 0.1 \
    --warmup_steps -1 \
    --do_train \
    --do_test \
    --setting run \
    --track_grad_norm 2 \
    --gradient_clip_val 1.0 \
    # --warmup_ratio 0.1 \