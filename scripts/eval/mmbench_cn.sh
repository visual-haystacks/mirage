#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
CKPT="mirage-llama3.1-8.3B-mmbench-cn"

python -m llava.eval.model_vqa_mmbench \
    --model-path "tsunghanwu/mirage-llama3.1-8.3B" \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/${CKPT}.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

mkdir -p playground/data/eval/mmbench_cn/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment $CKPT
