import torch
import torch.nn as nn
from utils import to_var, pad
import layers
from utils import to_var, SOS_ID, UNK_ID, EOS_ID, PAD_ID
import torch.nn.functional as F
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import os 

class DialoGPT(nn.Module):
    def __init__(self, config):
        super(DialoGPT, self).__init__()
        gpt2_config = GPT2Config(n_ctx=1024, n_embd=1024, n_layer=24, n_head=16)
        project_dir = config.dataset_dir.parent.parent
        pretrained_path = os.path.join(project_dir, 'src', 'model', 'pretrained', 'medum_ft.pkl')

        self.gpt2 = GPT2LMHeadModel(gpt2_config)

    def forward(self, x):
        return self.gpt2(x)

