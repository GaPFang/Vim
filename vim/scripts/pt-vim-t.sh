#!/bin/bash
# conda activate <your_env>
# cd <path_to_Vim>/vim;

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env vim/main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 25 --data-path /home/b11003/Vim/dataset/cifar100  --data-set CIFAR --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp
