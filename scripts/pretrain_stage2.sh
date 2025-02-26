#!/bin/bash
GPU_SETTINGS="localhost:0,1,2,3,4,5,6,7"
MASTER_PORT="19487"

deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version plain \
    --data_path playground/data/share-captioner_coco_lcs_sam_1246k_1107.json \
    --image_folder playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --pretrain_mm_mlp_adapter ./checkpoints/mirage_qformer_stage1_pretrain/mm_projector.bin \
    --pretrain_qformer ./checkpoints/mirage_qformer_stage1_pretrain/qformer.bin \
    --output_dir ./checkpoints/mirage_qformer_stage2_pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "mirage_qformer_stage2_pretrain" \
    --mm_reduce_token_method "qformer"
