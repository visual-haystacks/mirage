from PIL import Image
import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.utils import disable_torch_init



@torch.inference_mode()
def demo(model_path, image_paths, prompt, num_retrievals=1):
    # Model
    model_name = get_model_name_from_path(model_path)
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer, model, image_processor, _ = \
        load_pretrained_model(model_path=model_path, model_base=None, model_name=model_name, device="cuda")
    model.eval_mode = True


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
        with torch.inference_mode():
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
        with torch.inference_mode():
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="mirage_checkpoint")
    parser.add_argument("--max-num-retrievals", type=int, default=1)
    parser.add_argument("--image-folder", type=str, default="assets/example")
    parser.add_argument("--prompt", type=str, default="Here are a set of random images in my photo album. If you can find a cat, tell me what's the cat doing and what's its color.")
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_folder, image) for image in os.listdir(args.image_folder)]
    text_output, retrieval_output = demo(args.model_path, image_paths, args.prompt, args.max_num_retrievals)
    print('---Input---')
    print("Prompt:", args.prompt)
    print("Images:", image_paths)
    print('---Output---')
    print("Text Output:", text_output)
    retrieval_path = []
    for ret_output, image_path in zip(retrieval_output, image_paths):
        if ret_output == 1:
            retrieval_path.append(image_path)
    print("Retrieval Image:", retrieval_path)