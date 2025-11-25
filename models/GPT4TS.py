import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from transformers.models.llama.modeling_llama import LlamaModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.backbone = configs.backbone
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.backbone == 'gpt2':
            if configs.pretrain:
                self.llmmodel = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.llmmodel = GPT2Model(GPT2Config())
            self.llmmodel.h = self.llmmodel.h[:configs.gpt_layers]
            print("llmmodel = {}".format(self.llmmodel))

            if configs.freeze and configs.pretrain:
                for i, (name, param) in enumerate(self.llmmodel.named_parameters()):
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        if configs.backbone == 'llama2':
            self.llmmodel = LlamaModel.from_pretrained('meta-llama/Llama-2-7b-hf', output_attentions=True, output_hidden_states=True)
            self.llmmodel.layers = self.llmmodel.layers[:configs.gpt_layers]

            if configs.freeze and configs.pretrain:
                for i, (name, param) in enumerate(self.llmmodel.named_parameters()):
                    if 'input_layernorm' in name or 'post_attention_layernorm' in name: # or 'mlp' in name:
                        param.requires_grad = True
                    elif 'mlp' in name and configs.mlp == 1:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        

        for layer in (self.llmmodel, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.backbone == 'gpt2' or 'llama2':
            outputs = self.llmmodel(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
