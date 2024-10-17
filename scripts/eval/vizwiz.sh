#!/bin/bash
CKPT="mirage-llama3.1-8.3B-vizwiz"

python -m llava.eval.model_vqa_loader \
    --model-path "tsunghanwu/mirage-llama3.1-8.3B" \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode llama3

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}.json
