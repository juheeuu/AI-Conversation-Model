import random
import torch
from torch import nn
from torch.nn import functional as F
from .rnncells import StackedGRUCell, LSTMSACell
from .beam_search import Beam
from utils import to_var, SOS_ID, UNK_ID, EOS_ID, PAD_ID
from .encoder import PositionalEncoding
import pickle

class BaseRNNDecoder(nn.Module):
    def __init__(self):
        super(BaseRNNDecoder, self).__init__()

    def init_token(self, batch_size):
        x = to_var(torch.LongTensor([SOS_ID] * batch_size))
        return x

    def init_h(self, batch_size=None, zero=True, hidden=None):
        if hidden is not None:
            return hidden

        return to_var(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size
        else:
            batch_size = h.size(1)
            return batch_size

    def decode(self, out):
        if self.sample:
            x = torch.multinomial(self.softmax(out / self.temperature), 1).view(-1)
        else:
            _, x = out.max(dim=1)
        return x

    def forward(self):
        raise NotImplementedError

    def forward_step(self):
        raise NotImplementedError

    def embed(self, x):
        if self.training and self.word_drop > 0.0:
            if random.random() < self.word_drop:
                embed = self.embedding(to_var(x.data.new([UNK_ID] * x.size(0))))
            else:
                embed = self.embedding(x)
        else:
            embed = self.embedding(x)

        return embed

    def beam_decode(self, init_h=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(h=init_h)

        x = self.init_token(batch_size * self.beam_size)

        h = self.init_h(batch_size, hidden=init_h).repeat(1, self.beam_size, 1)

        batch_position = to_var(torch.arange(0, batch_size).long() * self.beam_size)

        score = torch.ones(batch_size * self.beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * self.beam_size, 0.0)
        score = to_var(score)

        beam = Beam(batch_size, self.hidden_size, self.vocab_size, self.beam_size, self.max_unroll, batch_position)

        for i in range(self.max_unroll):
            out, h = self.forward_step(x, h, encoder_outputs=encoder_outputs, input_valid_length=input_valid_length)
            log_prob = F.log_softmax(out, dim=1)

            score = score.view(-1, 1) + log_prob

            score, top_k_idx = score.view(batch_size, -1).topk(self.beam_size, dim=1)

            x = (top_k_idx % self.vocab_size).view(-1)

            beam_idx = top_k_idx / self.vocab_size  # [batch_size, beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            h = h.index_select(1, top_k_pointer)

            beam.update(score.clone(), top_k_pointer, x)  # , h)

            eos_idx = x.data.eq(EOS_ID).view(batch_size, self.beam_size)
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length


class DecoderRNN(BaseRNNDecoder):
    def __init__(self, vocab_size, embedding_size, hidden_size, rnncell=StackedGRUCell, num_layers=1,
                 dropout=0.0, word_drop=0.0, max_unroll=30, sample=True, temperature=1.0, beam_size=1):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.word_drop = word_drop
        self.max_unroll = max_unroll
        self.sample = sample
        self.beam_size = beam_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.rnncell = rnncell(num_layers, embedding_size, hidden_size, dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward_step(self, x, h, encoder_outputs=None, input_valid_length=None):
        x = self.embed(x)
        last_h, h = self.rnncell(x, h)
        out = self.out(last_h)

        return out, h

    def forward(self, inputs, init_h=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(inputs, init_h)

        x = self.init_token(batch_size)

        h = init_h

        if not decode:
            out_list = []
            seq_len = inputs.size(1)
            for i in range(seq_len):
                out, h = self.forward_step(x, h)

                out_list.append(out)
                x = inputs[:, i]

            return torch.stack(out_list, dim=1)
        else:
            x_list = []
            for i in range(self.max_unroll):
                out, h = self.forward_step(x, h)

                x = self.decode(out)
                x_list.append(x)

            return torch.stack(x_list, dim=1)


class DecoderSARNN(BaseRNNDecoder):
    def __init__(self, vocab_size, user_size, embedding_size, hidden_size, num_layers=1,
                 dropout=0.0, word_drop=0.0, max_unroll=30, sample=True, temperature=1.0, beam_size=1):
        super(DecoderSARNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.word_drop = word_drop
        self.max_unroll = max_unroll
        self.sample = sample
        self.beam_size = beam_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.user_embedding = nn.Embedding(user_size, embedding_size)

        self.rnncell = LSTMSACell(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward_step(self, x, user1_embed, user2_embed, h, encoder_outputs=None, input_valid_length=None):
        x = self.embed(x)
        hy, cy = self.rnncell(x, user1_embed, user2_embed, h)
        last_h = hy
        out = self.out(last_h)

        return out, (hy, cy)

    def forward(self, word_inputs, user_inputs, init_h=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(word_inputs, init_h)
        user_embedded = self.user_embedding(user_inputs)

        x = self.init_token(batch_size)
        h = init_h

        if not decode:
            out_list = []
            seq_len = word_inputs.size(1)
            for i in range(seq_len):
                out, h = self.forward_step(x, user_embedded[:, 0, :], user_embedded[:, 1, :], h)

                out_list.append(out)
                x = word_inputs[:, i]

            return torch.stack(out_list, dim=1)
        else:
            x_list = []
            for i in range(self.max_unroll):
                out, h = self.forward_step(x, user_embedded[:, 0, :], user_embedded[:, 1, :], h)

                x = self.decode(out)
                x_list.append(x)

            return torch.stack(x_list, dim=1)

    def beam_decode(self, init_h=None, user_inputs=None, encoder_outputs=None, input_valid_length=None, decode=False):
        batch_size = self.batch_size(user_inputs, h=init_h)
        user_embedded = self.user_embedding(user_inputs)

        x = self.init_token(batch_size * self.beam_size)

        hx, cx = init_h
        hx = hx.repeat(self.beam_size, 1)
        cx = cx.repeat(self.beam_size, 1)
        h = (hx, cx)
        user_emb_0 = user_embedded[:, 0, :].repeat(self.beam_size, 1)
        user_emb_1 = user_embedded[:, 1, :].repeat(self.beam_size, 1)

        batch_position = to_var(torch.arange(0, batch_size).long() * self.beam_size)

        score = torch.ones(batch_size * self.beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * self.beam_size, 0.0)
        score = to_var(score)

        beam = Beam(batch_size, self.hidden_size, self.vocab_size, self.beam_size, self.max_unroll, batch_position)

        for i in range(self.max_unroll):
            out, h = self.forward_step(x, user_emb_0, user_emb_1, h,
                                       encoder_outputs=encoder_outputs, input_valid_length=input_valid_length)
            log_prob = F.log_softmax(out, dim=1)

            score = score.view(-1, 1) + log_prob

            score, top_k_idx = score.view(batch_size, -1).topk(self.beam_size, dim=1)

            x = (top_k_idx % self.vocab_size).view(-1)

            beam_idx = top_k_idx / self.vocab_size
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            h = (h[0].index_select(0, top_k_pointer), h[1].index_select(0, top_k_pointer))
            # h = h.index_select(1, top_k_pointer)

            beam.update(score.clone(), top_k_pointer, x)

            eos_idx = x.data.eq(EOS_ID).view(batch_size, self.beam_size)
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length


class PTBDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                hidden_size, feedforward_hidden_size=2048, num_layers=12,
                num_heads=8, dropout=0.0, pretrained_wv_path=None, device=None, beam_size=1, max_seq_len=512):
        super(PTBDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers 
        self.num_heads = num_heads
        self.dropout = dropout 
        self.feedforward_hidden_size = feedforward_hidden_size
        self.device=device
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_ID)
        # if pretrained_wv_path is None:
        #     self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_ID)
        # else:
        #     with open(pretrained_wv_path, 'rb') as f:
        #         weight_tensor = to_var(torch.FloatTensor(pickle.load(f)))

        #     self.embedding = nn.Embedding.from_pretrained(weight_tensor, freeze=False)
        #     print("Load the wv Done")
        
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.dropoutLayer = nn.Dropout(dropout)

        self.decoder_stack = nn.ModuleList([
            AttentionRoutingLayer(hidden_size, num_heads, dropout, feedforward_hidden_size, device=device)
            for _ in range(num_layers)])

        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, encoder_output, target_utterance, target_utterance_mask=None, generate=False):
        # attn_mask (hidden_size, hidden_size)
        
        embedded = self.dropoutLayer(self.pos_encoder(self.embedding(target_utterance))) # (batch_size, max_seq_len, hidden_size)
        embedded = embedded.transpose(0, 1) #(max_seq_len, batch_size, hidden_size)

        # dec_output = self.decoder(encoder_output, embedded, target_utterance_mask)
        # dec_output = dec_output.transpose(0, 1) # (batch_size, max_seq_len, hidden_size)

        for dec_layer in self.decoder_stack:
            dec_output = dec_layer(encoder_output, embedded, target_utterance_mask, generate=generate)

        dec_output = self.decoder_norm(dec_output)
        dec_output = dec_output.transpose(0, 1)

        return dec_output

    # def forward_step(self, encoder_output, prev_tokens): 












class AttentionRoutingLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, feedforward_hidden_size, attn_mask=None, device=None):
        super(AttentionRoutingLayer, self).__init__()
        self.hidden_size = hidden_size
        self.feedforward_hidden_size = feedforward_hidden_size
        self.num_heads = num_heads
        self.device = device
        self.attn_mask = self.make_mask(self.hidden_size)

        self.OcAttnLayer = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.OprevAttnLayer = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_size, feedforward_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
    
    def forward(self, Ec, Eprev, Eprev_mask=None, generate=False):
        Oc = self.OcAttnLayer(Eprev, Ec, Ec)[0]
        attn_mask =  None if generate else self.attn_mask
        Oprev = self.OprevAttnLayer(Eprev, Eprev, Eprev, attn_mask=attn_mask, key_padding_mask=Eprev_mask)[0]

        Omerge = 2 * Oc + Oprev

        output = Omerge + self.dropout(Eprev)
        output = self.norm1(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout2(output2)
        output = self.norm2(output)

        return output


    def make_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = torch.FloatTensor(mask).to(self.device)
        return mask