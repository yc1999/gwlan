#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main.py \
    --strategy ddp \
    --val_check_interval 1.0 \
    --train_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --valid_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --test_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --model_name_or_path wpm \
    --train_batch_size 32 \
    --max_seq_len 128 \
    --max_epochs 1000 \
    --gpus 1 \
    --accumulate_grad_batches 1 \
    --vocab_size 50005 \
    --d_model 512 \
    --learning_rate 5e-4 \
    --dropout 0.1 \
    --warmup_ratio 0.1 \
    --do_test \
    --setting debug \
    --do_train \
    --track_grad_norm 2 \
    --gradient_clip_val 1.0 \
    # --test_dir ./dataset/wmt14_test_data/bi_context_raw_data/test.en2de.de.out.bi_context \
    # --ckpt_path "/home/yc21/project/gwlan/save/tmp/seed: 42 - epochs: 1000 - gpus: 1 - train_bacth_size: 64 - accumulate_grad_batches: 1/epoch=32--valid_acc=0.8854.ckpt" \