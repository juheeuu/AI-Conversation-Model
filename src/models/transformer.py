import torch
import torch.nn as nn
from utils import to_var, pad
import layers
from utils import to_var, SOS_ID, UNK_ID, EOS_ID, PAD_ID
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    """
    ref: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/position.py
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config 
        self.device = self.config.device 

        self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=self.config.pad_id)
        self.pos_embedding = PositionalEmbedding(d_model=config.embedding_size, max_len=config.max_seq_len)

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads, 
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.feedforward_hidden_size,
            dropout=config.dropout 
            )

        self.linear = nn.Linear(config.d_model, config.vocab_size)

        self.apply(self._initailze) 


    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight)
    
    
    def forward(self, input_utterances, input_mask, target_utterance, target_mask):

        enc_embed = self.tok_embedding(input_utterances) + self.pos_embedding(input_utterances)
        dec_embed = self.tok_embedding(target_utterance) + self.pos_embedding(target_utterance)

        enc_embed = torch.einsum('ijk->jik', enc_embed)
        dec_embed = torch.einsum('ijk->jik', dec_embed)

        src_key_padding_mask = input_mask
        tgt_key_padding_mask = target_mask
        memory_key_padding_mask = input_mask 
        tgt_mask = self.transformer.generate_square_subsequent_mask(target_utterance.size(1)).to(self.device)


        outputs = self.transformer(src=enc_embed, tgt=dec_embed, 
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_mask=tgt_mask)


        outputs = self.linear(outputs)
        outputs = torch.einsum('ijk->jik', outputs)

        return outputs