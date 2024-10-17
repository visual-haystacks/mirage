#!/bin/bash
GPU_SETTINGS="localhost:1,2,3,4,5,6"

MASTER_PORT="15999"
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=DEBUG deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version llama3 \
    --data_path ./gen_dataset/instruction_tuning_dataset/llava_raw.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-llama31/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/MIRAGE_20240916_lora_llava_llama3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name 20240916_reorg_lora_llava_llama3
