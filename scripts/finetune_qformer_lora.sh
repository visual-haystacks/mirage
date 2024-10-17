#!/bin/bash
GPU_SETTINGS="localhost:0,1,2,3,4,5,6,7"
MASTER_PORT="19487"

deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version llama3 \
    --data_path ./playground/data/mirage_ft.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/mirage_qformer_stage3_pretrain/mm_projector.pth \
    --pretrain_qformer ./checkpoints/mirage_qformer_stage3_pretrain/qformer.pth \
    --pretrain_retriever ./checkpoints/mirage_qformer_stage3_pretrain/retriever.pth \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/mirage_qformer_ft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name mirage_qformer_ft \
    --mm_reduce_token_method qformer_query_aware \
    --apply_retriever True \
    --tune_retriever False
