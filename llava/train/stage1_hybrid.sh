#!/bin/bash

# Hybrid Vision Tower训练脚本
# 融合DINOv2和SigLIP编码器
# vision_tower格式: hybridmodel-facebook/dinov2-base-&&&-siglip/CLIP-ViT-SO400M-14-384

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /nfs/AI/VideoEnhancement/dingfu/vg/LLaVA/playground/blip_laion_cc_sbu_558k.json \
    --image_folder /nfs/AI/VideoEnhancement/dingfu/vg/LLaVA/playground/data_zips/llava_pretrain/images \
    --vision_tower "hybridmodel-facebook/dinov2-base-&&&-ViT-SO400M-14-SigLIP" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-hybrid-dinov2-siglip-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True

