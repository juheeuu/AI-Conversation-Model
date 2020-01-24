import torch
import torch.nn as nn
from utils import to_var, pad
import layers

class PretrainingBased(nn.Module):
    def __init__(self, config):
        super(PretrainingBased, self).__init__()
        self.config = config 
        self.encoder = layers.PTBEncoder(
            config.vocab_size, config.embedding_size, config.encoder_hidden_size,
            config.feedforward_hidden_size, config.num_layer, config.num_heads, config.dropout, 
            pretrained_wv_path=config.pretrained_wv_path)
        self.decoder = layers.PTBDecoder(
            config.vocab_size, config.embedding_size, config.encoder_hidden_size,
            config.feedforward_hidden_size, config.num_layer, config.num_heads, config.dropout, 
            pretrained_wv_path=config.pretrained_wv_path
        )

        if config.tie_embedding:
            self.decoder.embedding = self.decoder.embedding

        self.linear = nn.Linear(config.encoder_hidden_size, config.vocab_size)
    
    def forward(self, input_utterances, input_utterances_mask, 
                target_utterance, target_utterance_mask):
        encoder_outputs = self.encoder(input_utterances, input_utterances_mask)
        decoder_outputs = self.decoder(encoder_outputs, input_utterances_mask, target_utterance, target_utterance_mask)
        outputs = self.linear(decoder_outputs)

        return outputs

    def generate(self):
        pass 
