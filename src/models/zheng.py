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

        self.config = config

        gpt_config = OpenAIGPTConfig().from_pretrained('openai-gpt')
        setattr(gpt_config, 'users', config.users)
        setattr(gpt_config, 'user_size', config.user_size)

        if config.pretrained:
            transformer = TransformerModule(gpt_config).from_pretrained('openai-gpt', config=gpt_config)
        else:
            transformer = TransformerModule(gpt_config)

        # for DataParallel 
        model_to_resize = transformer.module if hasattr(transformer, "module") else transformer
        model_to_resize.resize_token_embeddings(config.vocab_size)
        self.transformer = model_to_resize

        self.linear = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        # tie weights
        self.linear.weight = self.transformer.tokens_embed.weight

    def encode(self, prev, prev_mask, user_ids=None):
        return self.transformer(prev, prev_mask, user_ids=user_ids)
    
    def decode(self, x, x_mask, enc_hidden, prev_mask, user_ids=None):
        return self.linear(self.transformer(x, x_mask, enc_hidden, prev_mask, user_ids=user_ids))
    
    def forward(self, x, x_mask, prev, prev_mask, x_user_ids=None, prev_user_ids=None):
        enc_hidden = self.transformer(prev, prev_mask, user_ids=prev_user_ids)
        lm_output = self.linear(enc_hidden)
        conv_output = self.linear(self.transformer(x, x_mask, enc_hidden, prev_mask, user_ids=x_user_ids))

        return lm_output, conv_output
    
    def beam_generate(self, prev, prev_mask, prev_user_ids, x_user_ids, sos_id, pad_id, eos_id):

        batch_size = prev.size(0)
        max_seq_len = prev.size(1)
        beam_size = self.config.beam_size
        vocab_size = self.config.vocab_size
        length_penalty = 1.0

        enc_hidden = self.encode(prev, prev_mask, prev_user_ids) # (batch_size, max_seq_len, hidden_size)
        enc_hidden = enc_hidden.repeat(1, beam_size, 1) # (batch_size, max_seq_len * beam_size, hidden_size)
        enc_hidden = enc_hidden.contiguous().view(batch_size * beam_size, max_seq_len, -1) # (batch_size * beam_size, max_seq_len, hidden_size)
        
        prev_mask = prev_mask.repeat(1, beam_size).contiguous().view(-1, max_seq_len) # (batch_size * beam_size, max_seq_len)

        if x_user_ids is not None:
            x_user_ids = x_user_ids.repeat(1, beam_size).contiguous().view(-1, max_seq_len) # (batch_size * beam_size, max_seq_len)

        input_ids = torch.LongTensor([[sos_id]] * batch_size * beam_size).to(self.config.device) # (batch_size * beam_size, 1)

        batch_position = (torch.arange(0, batch_size).long() * beam_size).to(self.config.device)

        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=prev.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        done = [False for _ in range(batch_size)]

        generated_hyps = [
            BeamHypotheses(beam_size, max_seq_len, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        cur_len = 1

        while cur_len < max_seq_len:
            if x_user_ids is not None: 
                x_user_id = x_user_ids[...,:cur_len]
            scores = self.decode(input_ids, None, enc_hidden, prev_mask, x_user_id)[:,-1,:] # (batch_size * beam_size, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # (batch_size * beam_size, vocab_size)
            assert scores.size() == (batch_size * beam_size, vocab_size)

            _scores = beam_scores[:, None].expand_as(scores) + scores
            _scores = _scores.view(batch_size, beam_size * vocab_size)  # (batch_size, beam_size * vocab_size)
            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):
                                # if we are done with this sentence
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item()
                )
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= beam_size
                    ), "Batch can only be done if at least {} beams have been generated".format(beam_size)
                    assert (
                        eos_id is not None and pad_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_id, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # add to generated hypotheses if end of sentence or last iteration
                    if word_id.item() == eos_id:
                        generated_hyps[batch_idx].add(
                            input_ids[batch_idx * beam_size + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        # add next predicted word if it is not eos_token
                        next_sent_beam.append((score, word_id, batch_idx * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break
            
                # update next beam content
                assert len(next_sent_beam) == beam_size, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (batch_idx + 1)

            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            cur_len = cur_len + 1

            if all(done):
                break

        for batch_idx in range(batch_size):
            # Add all open beam hypothesis to generated_hyps
            if not done[batch_idx]:
                for idx, score in zip(next_words[batch_idx], next_scores[batch_idx]):
                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size
                    generated_hyps[batch_idx].add(
                        input_ids[batch_idx * beam_size + beam_id, :cur_len].clone(), score.item()
                    )

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.beams, key=lambda x: x[0])[1]
            sent_lengths[i] = len(best_hyp)
            best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_seq_len)
            decoded = input_ids.new(batch_size, sent_max_len).fill_(pad_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_seq_len:
                    decoded[i, sent_lengths[i]] = eos_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_seq_len for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded 



class TransformerModule(OpenAIGPTPreTrainedModel):
    def __init__(self, config): 
        super(TransformerModule, self).__init__(config)
        self.config = config

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
        if config.users:
            self.user_embed = nn.Embedding(config.user_size, embedding_size, padding_idx=0)

        self.drop = nn.Dropout(embed_dropout)
        self.h = nn.ModuleList(
            [TransformerBlock(embedding_size, n_heads, dropout,
                            attn_dropout, ff_dropout, max_seq_len) for _ in range(num_layers)])

        if pretrained: 
            self._init_weights_for_not_pretrained()
        
    def _init_weights_for_not_pretrained(self):
        nn.init.normal_(self.tokens_embed.weight, std=0.02)
        nn.init.normal_(self.positions_embed.weight, std=0.02)
        if self.config.users:
            nn.init.normal_(self.user_embed.weight, std=0.02)

    def forward(self, x, x_mask=None, enc_hidden=None, enc_hidden_mask=None, user_ids=None):
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

        if user_ids is not None and self.config.users: 
            user_embed = self.user_embed(user_ids)
            hidden_states = hidden_states + user_embed

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


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
