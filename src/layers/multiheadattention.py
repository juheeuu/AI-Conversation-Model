import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import to_var, PAD_ID
import pickle
import math


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, crossLayer=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        if crossLayer:
            self.query = Conv1D(n_state, nx)
            self.key = Conv1D(n_state, nx)
            self.value = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, masked=True):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)

        if masked:
            b = self.bias[:, :, ns - nd : ns, :ns]
            w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            # print(w.shape)
            # print(attention_mask)
            # print(attention_mask.shape)
            w = w + attention_mask
            # print(w)
            # exit()

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # (batch, seq_len, n_head, head_features) 
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x=None, query=None, key=None, value=None, attention_mask=None, head_mask=None, masked=True):
        
        if (query is not None) and (key is not None) and (value is not None):  
            query = self.query(query) # (batch, max_seq_len, hidden_size)
            key = self.key(key) # (batch, max_seq_len, hidden_size)
            value = self.value(value) # (batch, max_seq_len, hidden_size)

            query = self.split_heads(query) # (batch, head, head_features, seq_length)
            key = self.split_heads(key, k=True) # (batch, head, head_features, seq_length)
            value = self.split_heads(value) # (batch, head, head_features, seq_length)
        else:
            x = self.c_attn(x)
            query, key, value = x.split(self.split_size, dim=2)
            query = self.split_heads(query)
            key = self.split_heads(key, k=True)
            value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, masked=masked)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return a


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
