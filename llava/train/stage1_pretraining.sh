#!/bin/bash

# The deepspeed command will automatically use all 8 visible GPUs.
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    # ✅ Make sure this points to your Dinov2 model
    --vision_tower facebook/dinov2-base \
    # ✅ Make sure this points to the projector you trained in Stage 1
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-dinov2-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-dinov2-finetune \
    --num_train_epochs 1 \
    # ✅ Increase this to leverage A100 memory. Start with 24 or 32 and monitor GPU memory usage.
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    # ✅ With 8 GPUs, a gradient accumulation of 1 is fine.
    # Your global batch size will be 32 * 8 = 256.
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb