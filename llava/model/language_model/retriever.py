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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.generation.utils import GenerateOutput

# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM



class RetrieverConfig(LlamaConfig):
    model_type = "retriever"


class RetrieverModel(LlamaModel):
    config_class = RetrieverConfig

    def __init__(self, config: LlamaConfig):
        super(RetrieverModel, self).__init__(config)


class RetrieverForCausalLM(LlamaForCausalLM):
    config_class = RetrieverConfig

    def __init__(self, config):
        super(RetrieverForCausalLM, self).__init__(config)
        self.model = RetrieverModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

AutoConfig.register("retriever", RetrieverConfig)
AutoModelForCausalLM.register(RetrieverConfig, RetrieverForCausalLM)
