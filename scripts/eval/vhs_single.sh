#!/bin/bash
# Define the image counts
declare -a image_counts=("oracle" 2 3 5 10 20 50 100 500 1000 10000)

# Define the base paths
test_file_base="playground/data/eval/visual_haystacks/VHs_qa/single_needle/VHs_large"
image_root="playground/data/eval/visual_haystacks/coco"
output_dir_base="mirage_vhs_single_output"


# Loop over the image counts
for i in "${!image_counts[@]}"; do
    # Run oracle first
    val=${image_counts[$i]}
    test_file="${test_file_base}/visual_haystack_${val}.json"
    output_dir="${output_dir_base}/${val}_images"

    mkdir -p $output_dir

    if [[ "$val" == "oracle" ]] || [[ "$val" -le 100 ]]; then
        # Execute the python script
        python -m llava.eval.model_vqa_vhs \
            --model-path "tsunghanwu/mirage-llama3.1-8.3B" \
            --test-file "$test_file" \
            --image-folder "$image_root" \
            --output-dir "$output_dir" \
            --max-num-retrievals 1
    else
        python -m llava.eval.model_vqa_vhs \
            --model-path "tsunghanwu/mirage-llama3.1-8.3B" \
            --test-file "$test_file" \
            --image-folder "$image_root" \
            --output-dir "$output_dir" \
            --max-num-retrievals 1 \
            --quick_mode
    fi
done