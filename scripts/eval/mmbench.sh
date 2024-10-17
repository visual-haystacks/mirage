#!/bin/bash

SPLIT="mmbench_dev_20230712"
CKPT="mirage-llama3.1-8.3B-mmbench-en"

python -m llava.eval.model_vqa_mmbench \
    --model-path "tsunghanwu/mirage-llama3.1-8.3B" \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
