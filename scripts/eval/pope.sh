#!/bin/bash
CKPT="mirage-llama3.1-8.3B-pope"

python -m llava.eval.model_vqa_loader \
    --model-path "tsunghanwu/mirage-llama3.1-8.3B" \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode llama3

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco_pope/ \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${CKPT}.jsonl
