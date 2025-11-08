#!/bin/bash

# MRF Vision Tower Stage2 微调脚本
# 使用Stage1预训练的mm_projector进行指令微调

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower "hybridmodel-facebook/dinov2-base-&&&-siglip/CLIP-ViT-B-16" \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-hybrid-dinov2-siglip-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-mrf-stage2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
