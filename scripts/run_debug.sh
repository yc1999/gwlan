#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,7

python main.py \
    --train_dataset_ratio 1.0 \
    --strategy ddp \
    --check_val_every_n_epoch 1 \
    --train_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --valid_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --test_dir ./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.de.out.bi_context \
    --model_name_or_path wpm \
    --train_batch_size 10 \
    --max_seq_len 128 \
    --max_epochs 1000 \
    --gpus 2 \
    --accumulate_grad_batches 1 \
    --vocab_size 50005 \
    --d_model 512 \
    --learning_rate 5e-5 \
    --dropout 0.1 \
    --warmup_ratio 0.1 \
    --scheduler linear \
    --do_test \
    --setting debug \
    --do_train \
    --gradient_clip_val 1.0 \
    # --max_steps -1  \
    # --track_grad_norm 2 \
    # --max_epochs 1000 \
    # --test_dir ./dataset/wmt14_test_data/bi_context_raw_data/test.en2de.de.out.bi_context \
    # --val_check_interval 2.0 \
    # --ckpt_path "/home/yc21/project/gwlan/save/tmp/seed: 42 - epochs: 1000 - gpus: 1 - train_bacth_size: 64 - accumulate_grad_batches: 1/epoch=32--valid_acc=0.8854.ckpt" \