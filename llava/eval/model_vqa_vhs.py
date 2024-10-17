from PIL import Image
import argparse
import torch
import os
import json
import nltk
import random
import numpy as np
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.utils import disable_torch_init


@torch.inference_mode()
def generate(prompt, image_paths, tokenizer, model, image_processor, num_retrievals=1):

    conv = conv_templates["llama3"].copy()
    clip_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.to(dtype=torch.float16)
        clip_images.append(image_tensor)

    qformer_text_input = tokenizer(prompt, return_tensors='pt')["input_ids"].to(model.device)

    N = len(clip_images)
    img_str = DEFAULT_IMAGE_TOKEN * N + "\n"
    inp = img_str + prompt

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    N = len(clip_images)
    tokenizer.pad_token_id = 128002
    if N <= 200:
        batch_clip_imaegs = [torch.stack(clip_images).to(model.device)]
        output_ret, output_ids = model.generate(
            input_ids,
            pad_token_id=tokenizer.pad_token_id,
            clip_images=batch_clip_imaegs,
            qformer_text_input=qformer_text_input,
            relevance=None,
            num_retrieval=num_retrievals,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True)
    else:
        # batch size is too large, split into smaller batches
        batch_clip_imaegs = [torch.stack(clip_images)]
        output_ret, output_ids = model.batch_generate(
            input_ids,
            clip_images=batch_clip_imaegs,
            pad_token_id=tokenizer.pad_token_id,
            qformer_text_input=qformer_text_input,
            relevance=None,
            num_retrieval=num_retrievals,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True)

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if not isinstance(output_ret[0], list):
        output_ret[0] = output_ret[0].tolist()
    return output_text, output_ret[0]


def eval(dirname):
    all_files = [f for f in os.listdir(dirname) if f.endswith(".json")]
    data = [json.load(open(os.path.join(dirname, f), "r")) for f in all_files]

    acc_avgs = []
    comp_avgs = []
    recall_avgs = []
    precision_avgs = []
    N = len(data)
    all_indices = list(range(N))

    # Bootstraping Evaluation
    for _ in range(100):
        correct, compliance, total = 0, 0, 0
        fp, tp, fn = 0, 0, 0
        indices = np.random.choice(all_indices, N, replace=True)
        for i in indices:
            gt = data[i]["conversations"][1]["value"]
            pred = data[i]["result"]["response"]
            # Compute VQA metrics
            if "yes" in nltk.word_tokenize(
                pred.lower()
            ) or "no" in nltk.word_tokenize(pred.lower()):
                compliance += 1
            if gt in nltk.word_tokenize(pred.lower()):
                correct += 1
            total += 1
            # Compute retrieval metrics
            gt_ret = data[i]["result"]["retrieval_ground_truth"]
            pred_ret = data[i]["result"]["retrieval_output"]
            for pred, gt in zip(pred_ret, gt_ret):
                if pred == 1 and gt == 1:
                    tp += 1
                elif pred == 1 and gt == 0:
                    fp += 1
                elif pred == 0 and gt == 1:
                    fn += 1
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        acc_avg = correct / total
        comp_avg = compliance / total
        acc_avgs.append(acc_avg)
        comp_avgs.append(comp_avg)
        recall_avgs.append(recall)
        precision_avgs.append(precision)
    acc_avg = np.mean(acc_avgs)
    acc_std = np.std(acc_avgs)
    comp_avg = np.mean(comp_avgs)
    comp_std = np.std(comp_avgs)
    recall_avg = np.mean(recall_avgs)
    recall_std = np.std(recall_avgs)
    precision_avg = np.mean(precision_avgs)
    precision_std = np.std(precision_avgs)
    print("[Retrieval Metrics]")
    print(f"Recall: {recall_avg:.4f} ± {recall_std:.4f}, Precision: {precision_avg:.4f} ± {precision_std:.4f}")
    print("[VQA Metrics]")
    print(f"Compliance: {comp_avg:.4f} ± {comp_std:.4f}, Accuracy: {acc_avg:.4f} ± {acc_std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="mirage_checkpoint")
    parser.add_argument("--max-num-retrievals", type=int, default=1)
    parser.add_argument("--image-folder", type=str, default="coco")
    parser.add_argument("--test-file", type=str, default="visual_haystack_oracle.json")
    parser.add_argument("--output-dir", type=str, default="vhs_single_oracle")
    parser.add_argument("--quick_mode", action="store_true")
    args = parser.parse_args()

    # Model
    model_name = get_model_name_from_path(args.model_path)
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer, model, image_processor, _ = \
        load_pretrained_model(model_path=args.model_path, model_base=None, model_name=model_name, device="cuda")
    model.eval_mode = True

    test_data = json.load(open(args.test_file, "r"))
    if args.quick_mode:
        random.shuffle(test_data)
        test_data = test_data[::3]
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, entry in enumerate(tqdm(test_data)):
        prompt = "You are given a set of images. Please answer the following question in Yes or No: " + entry["conversations"][0]['value']
        pos_img_paths = [os.path.join(args.image_folder, rel_path) for rel_path in entry["pos_image"]]
        neg_img_paths = [os.path.join(args.image_folder, rel_path) for rel_path in entry["neg_image"]]
        
        image_paths = pos_img_paths + neg_img_paths
        random.shuffle(image_paths)
        gt_ret = []
        for img in image_paths:
            if img in pos_img_paths:
                gt_ret.append(1)
            else:
                gt_ret.append(0)
        response, output_ret = generate(prompt, image_paths, tokenizer, model, image_processor, args.max_num_retrievals)
        entry["result"] = {
            "image_paths": image_paths,
            "retrieval_ground_truth": gt_ret,
            "retrieval_output": output_ret,
            "response": response
        }
        with open(os.path.join(args.output_dir, f"{idx}.json"), "w") as f:
            json.dump(entry, f)
    print(f"Evaluating on {args.output_dir}")
    eval(args.output_dir)