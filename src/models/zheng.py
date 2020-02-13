import torch
import torch.nn as nn
from utils import to_var, pad
import layers
from utils import to_var, SOS_ID, UNK_ID, EOS_ID, PAD_ID
import torch.nn.functional as F
import torch.nn as nn
from transformers import OpenAIGPTModel, OpenAIGPTConfig, OpenAIGPTPreTrainedModel

class ZHENG(nn.Module):
    def __init__(self, config):
        super(ZHENG, self).__init__()

        if config.pretrained:
            gpt_config = OpenAIGPTConfig.from_pretrained('openai-gpt')
            transformer = TransformerModule(gpt_config).from_pretrained('openai-gpt')
            # for DataParallel
            model_to_resize = transformer.module if hasattr(transformer, "module") else transformer
            model_to_resize.resize_token_embeddings(config.vocab_size)
            self.transformer = model_to_resize
        else:
            self.transformer = TransformerModule(config)

        self.linear = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        # tie weights
        self.linear.weight = self.transformer.tokens_embed.weight
    
    def forward(self, x, x_mask, prev, prev_mask):
        enc_hidden = self.transformer(prev, prev_mask)
        lm_output = self.linear(enc_hidden)
        conv_output = self.linear(self.transformer(x, x_mask, enc_hidden, prev_mask))

        return lm_output, conv_output

class TransformerModule(OpenAIGPTPreTrainedModel):
    def __init__(self, config): 
        super(TransformerModule, self).__init__(config)

        pretrained = True if isinstance(config, OpenAIGPTConfig) else False

        embedding_size = config.n_embd if pretrained else config.embedding_size
        max_seq_len = config.n_positions if pretrained else config.max_seq_len
        n_heads = config.n_head if pretrained else config.n_heads 
        embed_dropout = config.embd_pdrop if pretrained else config.embed_dropout 
        dropout = config.embd_pdrop if pretrained else config.dropout
        attn_dropout = config.attn_pdrop if pretrained else config.attn_dropout
        ff_dropout = config.resid_pdrop if pretrained else config.ff_dropout
        num_layers = config.n_layer if pretrained else config.num_layers

        self.tokens_embed = nn.Embedding(config.vocab_size, embedding_size)
        self.positions_embed = nn.Embedding(max_seq_len, embedding_size, padding_idx=0)

        self.drop = nn.Dropout(embed_dropout)
        self.h = nn.ModuleList(
            [TransformerBlock(embedding_size, n_heads, dropout,
                            attn_dropout, ff_dropout, max_seq_len) for _ in range(num_layers)])

        if pretrained: 
            self._init_weights_for_not_pretrained()
        
    def _init_weights_for_not_pretrained(self):
        nn.init.normal_(self.tokens_embed.weight, std=0.02)
        nn.init.normal_(self.positions_embed.weight, std=0.02)

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

    def get_input_embeddings(self):
        return self.tokens_embed
    
    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

    
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

