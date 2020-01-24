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
                 hidden_size, feedforward_hidden_size=2048, num_layers=12,
                 num_heads=8, dropout=0.0, pretrained_wv_path=None):
        super(PTBEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.dropout = dropout 
        self.feedforward_hidden_size = feedforward_hidden_size

        if pretrained_wv_path is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_ID)
        else:
            with open(pretrained_wv_path, 'rb') as f:
                weight_tensor = to_var(torch.FloatTensor(pickle.load(f)))

            self.embedding = nn.Embedding.from_pretrained(weight_tensor, freeze=False)
            print("Load the wv Done")
        
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.dropoutLayer = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, feedforward_hidden_size, dropout)
        encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, inputs, inputs_mask):
        """
         inputs : (batch_size, max_seq_len, hidden_size)
         inputs_mask : (batch_size, max_seq_len) BOOL 
        """
        # attn_mask (hidden_size, hidden_size)
        if self.attn_mask == None: 
            self.attn_mask = self.make_mask(self.hidden_size)

        embedded = self.dropoutLayer(self.pos_encoder(self.embedding(inputs))) # (batch_size, max_seq_len, hidden_size)

        embedded = embedded.transpose(0, 1) #(max_seq_len, batch_size, hidden_size)
        inputs_mask = inputs_mask.transpose(0, 1) #(max_seq_len, batch_size, hidden_size)

        enc_output = self.encoder(embedded, mask=self.attn_mask, key_padding_mask=inputs_mask)
        # (max_seq_len, batch_size, hidden_size)
        enc_output = enc_output.transpose(0, 1) # (batch_size, max_seq_len, hidden_size)

        return enc_output

    def make_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
