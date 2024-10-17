import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
from typing import Dict, Sequence
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict, image_folder, tokenizer, clip_image_processor, model_config):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.clip_image_processor = clip_image_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        image_files = self.list_data_dict[i]['image']
        # handle cases for single image
        if type(image_files) != list:
            image_files = [image_files]
        clip_images = []
        for image_file in image_files:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            clip_image = process_images([image], self.clip_image_processor, self.model_config)[0]
            clip_images.append(clip_image)

        qs = sources[0]['conversations'][0]['value']
        qformer_text_input = self.tokenizer(qs, return_tensors='pt')["input_ids"][0]
        N = len(clip_images)
        img_str = DEFAULT_IMAGE_TOKEN * N + "\n"
        qs = img_str + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        relevant_images = torch.tensor(sources[0]['relevant_images'])
        return dict(clip_images=torch.stack(clip_images), 
                    input_ids=input_ids, 
                    qformer_text_input=qformer_text_input,
                    relevant_images=relevant_images)


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, qformer_text_input = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "qformer_text_input"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        qformer_text_input = torch.nn.utils.rnn.pad_sequence(
            qformer_text_input,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        qformer_text_input = qformer_text_input[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            qformer_text_input=qformer_text_input,
            clip_images=[instance['clip_images'] for instance in instances],
            relevant_images=[instance['relevant_images'] for instance in instances]
        )

        return batch



# DataLoader
def create_data_loader(test_data, image_folder, tokenizer, clip_image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = LazySupervisedDataset(test_data, image_folder, tokenizer, clip_image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=DataCollatorForSupervisedDataset(tokenizer))
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, clip_image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval_mode = True
    tokenizer.pad_token_id = 128002

    # Data
    test_data = json.load(open(args.test_file, "r"))
    test_data = get_chunk(test_data, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(test_data, args.image_folder, tokenizer, clip_image_processor, model.config)
    
    # whether passing oracle relevant images to LMMs or requires retrieval
    oracle_case = False

    for batch, line in tqdm(zip(data_loader, test_data), total=len(test_data)):
        
        idx = line["id"]
        cur_prompt = line['conversations'][0]['value']

        input_ids = batch['input_ids'].to(device='cuda', non_blocking=True)
        clip_image_tensor = [batch['clip_images'][0].to(dtype=torch.float16, device='cuda', non_blocking=True)]
        qformer_text_input = batch['qformer_text_input'].to(device='cuda', non_blocking=True)
        if oracle_case is True:
            relevance = batch['relevant_images']
        else:
            relevance = None
        with torch.inference_mode():
            ret_results, output_ids = model.generate(
                input_ids,
                pad_token_id=tokenizer.pad_token_id,
                clip_images=clip_image_tensor,
                qformer_text_input=qformer_text_input,
                relevance=relevance,
                num_retrieval=args.max_num_retrievals,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        pred_relevance = ret_results[0]
        if isinstance(pred_relevance, list):
            pred_relevance = pred_relevance
        else:
            pred_relevance = pred_relevance.cpu().numpy().tolist()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "relevant": pred_relevance,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-v1.5-7b-retvqa-20240413")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--max-num-retrievals", type=int, default=3)
    parser.add_argument("--image-folder", type=str, default="playground/data/")
    parser.add_argument("--test-file", type=str, default="retvqa_test.json")
    parser.add_argument("--answers-file", type=str, default="./answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
