#!/bin/bash
GPU_SETTINGS="localhost:0,1,2,3,4,5,6,7"
MASTER_PORT="19487"

deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version plain \
    --data_path ./playground/data/stage1_pretraining.txt \
    --data_format wds \
    --max_steps 40000 \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/mirage_qformer_stage1_pretrain \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.005 \
    --warmup_steps 500 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "mirage_qformer_stage1_pretrain" \
    --mm_reduce_token_method "qformer" 
