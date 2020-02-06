import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import to_var, PAD_ID
import pickle
import math


class BaseRNNEncoder(nn.Module):
    def __init__(self):
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, 'rnn'):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError('no rnn selected')

    def init_h(self, batch_size=None, hidden=None):
        if hidden is not None:
            return hidden

        if self.use_lstm:
            return (to_var(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)),
                    to_var(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)))
        else:
            return to_var(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self):
        raise NotImplementedError


class EncoderRNN(BaseRNNEncoder):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, rnn=nn.GRU, num_layers=1, bidirectional=False,
                 dropout=0.0, bias=True, batch_first=True, pretrained_wv_path=None):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        if pretrained_wv_path is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_ID)
        else:
            with open(pretrained_wv_path, 'rb') as f:
                weight_tensor = to_var(torch.FloatTensor(pickle.load(f)))

            self.embedding = nn.Embedding.from_pretrained(weight_tensor, freeze=False)
            print("Load the wv Done")

        self.rnn = rnn(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                       bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, inputs, input_length, hidden=None):
        input_length_sorted, indices = input_length.sort(descending=True)
        input_length_sorted = input_length_sorted.data.tolist()

        inputs_sorted = inputs.index_select(0, indices)

        embedded = self.embedding(inputs_sorted)

        rnn_input = pack_padded_sequence(embedded, input_length_sorted, batch_first=self.batch_first)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)
        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=self.batch_first)

        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices), hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        return outputs, hidden


class ContextRNN(BaseRNNEncoder):
    def __init__(self, input_size, context_size, rnn=nn.GRU, num_layers=1, dropout=0.0,
                 bidirectional=False, bias=True, batch_first=True):
        super(ContextRNN, self).__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = self.context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = rnn(input_size=input_size, hidden_size=context_size, num_layers=num_layers, bias=bias,
                       batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, encoder_hidden, conversation_length, hidden=None):
        batch_size, seq_len, _ = encoder_hidden.size()

        conv_length_sorted, indices = conversation_length.sort(descending=True)
        conv_length_sorted = conv_length_sorted.data.tolist()
        encoder_hidden_sorted = encoder_hidden.index_select(0, indices)

        rnn_input = pack_padded_sequence(encoder_hidden_sorted, conv_length_sorted, batch_first=True)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)

        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices), hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        return outputs, hidden

    def step(self, encoder_hidden, hidden):
        batch_size = encoder_hidden.size(0)
        encoder_hidden = torch.unsqueeze(encoder_hidden, 1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)

        outputs, hidden = self.rnn(encoder_hidden, hidden)
        return outputs, hidden


class PTBEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, feedforward_hidden_size=2048, num_layers=12, max_seq_len=512,
                 num_heads=8, dropout=0.0, pretrained_wv_path=None, device=None, pad_token_idx=None, layer_norm_epsilon=1e-6):
        super(PTBEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.dropout = dropout 
        self.feedforward_hidden_size = feedforward_hidden_size
        self.device = device


        self.wte = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_idx)
        self.pte = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        # 여기서 n_ctx 가 뭐임 
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size, layer_norm_epsilon)

        
    def forward(self, input_ids, input_mask, position_ids, attn_mask=None):
        """
         inputs : (batch_size, max_seq_len)
         inputs_mask : (batch_size, max_seq_len) BOOL 
        """
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        device = input_ids.device
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attn_mask is not None:
            attn_mask = attn_mask.view(-1, input_shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        head_mask = [None] * self.config.n_layer

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states,  attention_mask=attention_mask, head_mask=head_mask[i]
            )

        hidden_states = self.ln_f(hidden_states)

        return hidden_states




        # embedded = self.dropoutLayer(self.pos_encoder(self.embedding(inputs))) # (batch_size, max_seq_len, embedded_size)

        # inputs = embedded.transpose(0, 1) #(max_seq_len, batch_size, hidden_size)

        # enc_output = self.encoder(inputs, mask=attn_mask, src_key_padding_mask=inputs_mask)
        # # (max_seq_len, batch_size, hidden_size)
        # enc_output = enc_output.transpose(0, 1) # (batch_size, max_seq_len, hidden_size)

        return enc_output
