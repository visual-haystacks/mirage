#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="mirage-llama3.1-8.3B-retvqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_retvqa \
        --model-path "tsunghanwu/mirage-llama3.1-8.3B" \
        --test-file ./playground/data/eval/retvqa/retvqa_test_mirage.json \
        --image-folder ./playground/data/eval/retvqa \
        --answers-file ./playground/data/eval/retvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llama3 &
done

wait

output_file=./playground/data/eval/retvqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/retvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done