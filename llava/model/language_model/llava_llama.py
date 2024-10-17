#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import heapq
import random
import numpy as np
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM



class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)



def flip_zeros(binary_arrays, max_flip_percentage=0.25):
    binary_arrays = [array.clone() for array in binary_arrays]
    for array in binary_arrays:
        # Calculate the random flip percentage for this array
        flip_percentage = random.uniform(0, max_flip_percentage)

        # Identify indices of all zeros
        zero_indices = (array == 0).nonzero(as_tuple=False).view(-1)
        
        # Determine the number of zeros to flip based on the random flip percentage
        num_to_flip = round(len(zero_indices) * flip_percentage)
        # Randomly select indices of zeros to flip
        if num_to_flip > 0:
            indices_to_flip = zero_indices[torch.randperm(len(zero_indices))[:num_to_flip]]
        
            # Flip the selected zeros to ones
            array[indices_to_flip] = 1
    return binary_arrays


def process_relevance_outputs(confidences, top_k=1):
    # Get the confidence scores from the outputs

    # confidence: 1, N

    # Step 1: Filter samples with confidence >= 0.5
    valid_num = (confidences >= 0.5).sum()
    
    # Step 2 & 3: Check the number of valid samples
    if valid_num == 0:
        # No valid samples, use argmax on all samples to find the index with the highest confidence
        top_index = torch.argmax(confidences)
        relevance = torch.zeros_like(confidences).int()
        relevance[0, top_index] = 1
        relevance = relevance.cpu().tolist()
        assert np.sum(relevance) == 1
        return relevance
    else:
        # choose top min(5, valid_num) samples
        top_indices = torch.argsort(confidences, descending=True)[0, :top_k]
        relevance = torch.zeros_like(confidences).int()
        relevance[0, top_indices] = 1
        relevance = relevance.cpu().tolist()
        return relevance


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        clip_images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        relevance: Optional[torch.LongTensor] = None,
        qformer_text_input: Optional[torch.LongTensor] = None,
        epoch: Optional[float] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        relevance_output = None
        if inputs_embeds is None:
            ret = self.get_image_features(clip_images, qformer_text_input)
            retriever = self.get_model().get_retriever()
            if retriever is not None and ret is not None:
                image_features, question_embed = ret
                # context-free retriever
                relevance_output = retriever(image_features, question_embed, relevance)
                if epoch is not None:
                    if epoch < 0.6:
                        relevance_input = relevance.copy()
                    else:
                        if random.random() < 0.1: # use noises
                            relevance_input = flip_zeros(relevance, max_flip_percentage=0.15)
                        else:
                            relevance_input = relevance.copy()
                else:
                    relevance_input = relevance.copy()
            else:
                relevance_input = None
                if ret is not None:
                    image_features, question_embed = ret
                else:
                    image_features = None
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                image_features,
                image_sizes,
                relevance_input,
            )
        if getattr(self, "eval_mode", False) is True:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            return relevance_output, super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        clip_images: Optional[torch.Tensor] = None,
        qformer_text_input: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        relevance: Optional[torch.Tensor] = None,
        num_retrieval: Optional[int] = 1,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if clip_images is not None:
            image_features, question_embed = self.get_image_features(clip_images, qformer_text_input)
            if relevance is None:
                retriever = self.get_model().get_retriever()
                if retriever is not None:
                    # context-free retriever
                    relevance_output = retriever(image_features, question_embed, relevance)
                    confidences = torch.stack([output.detach()[:, 0] for output in relevance_output['outputs']])
                    relevance = process_relevance_outputs(confidences, top_k=num_retrieval)
                    del relevance_output
            
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                image_features,
                image_sizes=image_sizes,
                relevance=relevance,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return relevance, super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


    @torch.no_grad()
    def batch_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        clip_images: Optional[torch.Tensor] = None,
        qformer_text_input: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        relevance: Optional[torch.Tensor] = None,
        num_retrieval: Optional[int] = 1,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # relevance: [[0, 1, ...]] -->(1, N)
        # clip_images: [[N, C, H, W]] --> A list of N, C, H, W
        if clip_images is not None:
            # the batch size is too large, process each batch separately
            retriever = self.get_model().get_retriever()
            assert (relevance is not None) or (retriever is not None)
            if relevance is not None:
                image_features = []
                effective_clip_features = []
                N = len(relevance[0])
                for i in range(N):
                    if relevance[0][i] == 1:
                        effective_clip_features.append(clip_images[i])
                effective_image_features, _ = self.get_image_features(effective_clip_features.to(self.device), None, qformer_text_input)
                for i in range(N):
                    if relevance[0][i] == 0:
                        image_features.append(None)
                    else:
                        image_features.append(effective_image_features.pop(0))
                # image-features: a list of [[N, 32, 4096]]
                # question embed: a list of [[N, q_length, 4096]]

                
            else:
                # batch inference
                batch_size = 200
                all_scores = []
                all_features = []
                for i in range(0, len(clip_images[0]), batch_size):
                    # print(f"Processing batch {i} to {i + batch_size}", flush=True)
                    batch_image_features, question_embed = self.get_image_features(
                        [clip_images[0][i:i+batch_size].to(self.device)], qformer_text_input
                    )
                    relevance_output = retriever(batch_image_features, question_embed, None)
                    batch_image_features = batch_image_features[0].cpu()
                    scores = [output.detach()[:, 0].cpu() for output in relevance_output['outputs']]
                    all_scores.extend(scores[0])
                    all_features.extend(batch_image_features)

                all_scores = torch.stack(all_scores).unsqueeze(0)
                relevance = process_relevance_outputs(all_scores, top_k=num_retrieval)
                image_features = []
                for i in range(len(relevance[0])):
                    if relevance[0][i] == 1:
                        image_features.append(all_features[i].to(self.device, dtype=torch.float16))
                    else:
                        image_features.append(torch.rand(0, 4096).to(self.device, dtype=torch.float16))
                image_features = [image_features]
            
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                image_features,
                image_sizes=image_sizes,
                relevance=relevance,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return relevance, super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        clip_images = kwargs.pop("clip_images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        qformer_text_input = kwargs.pop("qformer_text_input", None)
        relevance = kwargs.pop("relevance", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if clip_images is not None:
            inputs['images'] = clip_images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if qformer_text_input is not None:
            inputs['qformer_text_input'] = qformer_text_input
        if relevance is not None:
            inputs['relevance'] = relevance
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
