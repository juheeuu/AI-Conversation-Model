import torch
import torch.nn as nn
from utils import to_var, pad
import layers
from utils import to_var, SOS_ID, UNK_ID, EOS_ID, PAD_ID
import torch.nn.functional as F
import torch.nn as nn

# TODO: Support huggingface OpenAI-GPT

class ZHENG(nn.Module):
    def __init__(self, config):
        super(ZHENG, self).__init__()

        self.transformer = TransformerModule(
            config.vocab_size, config.embedding_size, config.pad_id,
            config.max_seq_len, config.embed_dropout, config.n_heads,
            config.dropout, config.attn_dropout, config.ff_dropout,
            config.num_layers
        )

        self.linear = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.linear.weight = self.transformer.tokens_embed.weight
    
    def forward(self, x, x_maks, prev, prev_mask):
        enc_hidden = self.transformer(prev, prev_mask)
        lm_output = self.linear(enc_hidden)
        conv_output = self.linear(self.transformer(x, x_maks, enc_hidden, prev_mask))

        return lm_output, conv_output

class TransformerModule(nn.Module):
    def __init__(self, vocab_size, embedding_size, pad_id, max_seq_len, embed_dropout,
                n_heads, dropout, attn_dropout, ff_dropout, num_layers): 
        super(TransformerModule, self).__init__()
        self.tokens_embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.positions_embed = nn.Embedding(max_seq_len + 1, embedding_size, padding_idx=0)

        self.drop = nn.Dropout(embed_dropout)
        self.h = nn.ModuleList(
            [TransformerBlock(embedding_size, n_heads, dropout, attn_dropout, ff_dropout, max_seq_len) for _ in range(num_layers)])

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x, x_mask=None, enc_hidden=None, enc_hidden_mask=None):
        device = x.device

        x_shape = x.size()
        x = x.view(-1, x_shape[-1])

        pos_ids = torch.arange(x_shape[-1], dtype=torch.long, device=device)
        pos_ids = pos_ids.unsqueeze(0).view(-1, x_shape[-1])

        if x_mask is not None: 
            x_mask = x_mask.unsqueeze(1).unsqueeze(2)
            x_mask = x_mask.to(dtype=next(self.parameters()).dtype)  
            x_mask = (1.0 - x_mask) * -10000.0

        if enc_hidden_mask is not None: 
            enc_hidden_mask = enc_hidden_mask.unsqueeze(1).unsqueeze(2)
            enc_hidden_mask = enc_hidden_mask.to(dtype=next(self.parameters()).dtype)  
            enc_hidden_mask = (1.0 - enc_hidden_mask) * -10000.0

        inputs_embeds = self.tokens_embed(x)
        position_embeds = self.positions_embed(pos_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):
            hidden_states = block(hidden_states, x_mask, enc_hidden=enc_hidden, enc_hidden_mask=enc_hidden_mask)

        return hidden_states

    
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout, max_seq_len):
        super(TransformerBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(n_features)
        self.attn = layers.MultiheadAttention(n_features, n_heads, attn_dropout, ff_dropout, max_seq_len)
        self.ln_2 = nn.LayerNorm(n_features)
        self.mlp = layers.MLP(4 * n_features, n_features, ff_dropout)

    def forward(self, x, attention_mask=None, head_mask=None, enc_hidden=None, enc_hidden_mask=None):

        a = self.attn (
            self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=attention_mask, head_mask=head_mask
        )

        if enc_hidden is not None: 
            a += self.attn(self.ln_1(x), self.ln_1(enc_hidden), self.ln_1(enc_hidden),
                            attn_mask=enc_hidden_mask, head_mask=head_mask, qkv_same=False)

        x = x + a 
        m = self.mlp(self.ln_2(x))
        x = x + m 

        return x 

