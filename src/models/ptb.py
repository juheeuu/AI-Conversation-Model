import torch
import torch.nn as nn
from utils import to_var, pad
import layers
from utils import to_var, SOS_ID, UNK_ID, EOS_ID, PAD_ID
import torch.nn.functional as F


class PTB(nn.Module):
    def __init__(self, config):
        super(PTB, self).__init__()
        self.config = config 
        self.hidden_size = self.config.encoder_hidden_size
        self.device = self.config.device 
        
        self.encoder = layers.PTBEncoder(
            config.vocab_size, config.embedding_size, config.encoder_hidden_size,
            feedforward_hidden_size=config.feedforward_hidden_size, 
            num_layers=config.num_layers, 
            num_heads=config.num_heads, 
            dropout=config.dropout, 
            pretrained_wv_path=config.pretrained_wv_path,
            device=config.device
            )
        self.decoder = layers.PTBDecoder(
            config.vocab_size, config.embedding_size, config.encoder_hidden_size,
            feedforward_hidden_size=config.feedforward_hidden_size, 
            num_layers=config.num_layers, 
            num_heads=config.num_heads, 
            dropout=config.dropout, 
            pretrained_wv_path=config.pretrained_wv_path,
            device=config.device,
            beam_size=config.beam_size,
            max_seq_len=config.max_seq_len
            )

        self.linear = nn.Linear(config.encoder_hidden_size, config.vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        if config.tie_embedding:
            self.decoder.embedding.weight = self.encoder.embedding.weight

            # Encoder and Decoder share the same set of parameters  
            for en_layer, dec_layer in zip(self.encoder.encoder.layers, self.decoder.decoder_stack):
                dec_layer.OcAttnLayer = en_layer.self_attn
                dec_layer.OcPrevAttnLayer = en_layer.self_attn
                dec_layer.linear1.weight = en_layer.linear1.weight
                dec_layer.linear2.weight = en_layer.linear2.weight
                dec_layer.norm1.weight = en_layer.norm1.weight
                dec_layer.norm2.weight = en_layer.norm2.weight

            self.decoder.decoder_norm.weight = self.encoder.encoder.norm.weight



    
    
    def forward(self, input_utterances, input_utterances_mask, 
                target_utterance, target_utterance_mask):
        attn_mask_for_enc = self.make_mask(self.hidden_size).to(self.device)
        attn_mask_for_dec = attn_mask_for_enc.clone()
        encoder_outputs = self.encoder(input_utterances, input_utterances_mask, attn_mask=attn_mask_for_enc)
        decoder_outputs = self.decoder(encoder_outputs, target_utterance, target_utterance_mask, attn_mask=attn_mask_for_dec)
        outputs = self.linear(decoder_outputs)
        return outputs
    

    def make_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = torch.FloatTensor(mask)
        return mask

    def generate(self, input_utterances, input_utterances_mask):

        """
        Generate the response based on the input utterances 
        """

        encoder_outputs = self.encoder(input_utterances, input_utterances_mask)

        # Expand input to num beams 
        batch_size = input_utterances.size(0)
        beam_size = self.config.beam_size
        max_seq_len = self.config.max_seq_len
        vocab_size = self.config.vocab_size

        # start with SOS TOKEN
        init_seq = [SOS_ID]

        cur_len = 1

        input_ids = torch.LongTensor(batch_size * init_seq).to(self.config.device)
        input_ids = input_ids.unsqueeze(-1).unsqueeze(-1).expand(batch_size, beam_size, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * beam_size, cur_len) 

        # TODO:
        generated_hyps = [
            BeamHypotheses(beam_size, max_seq_len, early_stopping=False) for _ in range(batch_size)
        ]

        # is_done method
        # add method
        
        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=self.config.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * beam_size,)

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_seq_len: 
            outputs = self.linear(self.decoder(encoder_outputs, input_ids, generate=True)) # (batch_size * beam_size, cur_len, vocab_size)
            scores = outputs[:,-1,:] # (batch_size * beam_size, vocab_size)

            scores = F.log_softmax(scores, dim=-1) # (batch_size * beam_size, vocab_size)
            _scores = scores + beam_scores[:, None].expand_as(scores) # (batch_size * beam_size, vocab_size)
            _scores =  _scores.view(batch_size, beam_size * vocab_size)  # (batch_size, beam_size * vocab_size)
            next_scores, next_words = torch.topk(_scores, 2 * beam_size, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)

            next_batch_beam = []
            
            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            for batch_ex in range(batch_size):
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())

                if done[batch_ex]:
                    next_batch_beam.extend([(0, PAD_ID, 0)] * beam_size)  # pad the batch
                    continue

                next_sent_beam = []

                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    if word_id == EOS_ID or cur_len + 1 == max_seq_len:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * beam_size + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * beam_size + beam_id))
                    
                    # beam for next sentence is full...
                    if len(next_sent_beam) == beam_size:
                        break

                    # update the next beam contents 
                
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_seq_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, PAD_ID, 0)] * beam_size # pad the batch 
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (batch_ex + 1)


            # sanity check / prepare next batch
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

        tgt_len = input_ids.new(batch_size)
        best = []

        batch_loss = 0.0

        for i, hypothesis in enumerate(generated_hyps):
            seq_loss, best_hyp = max(hypothesis.hyp, key=lambda x: x[0])
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol\
            batch_loss += seq_loss
            best.append(best_hyp)
        
        # generate target batch
        decoded = input_ids.new(batch_size, tgt_len.max().item()).fill_(PAD_ID)
        for i, hypo in enumerate(best):
            decoded[i, : tgt_len[i] - 1] = hypo
            decoded[i, tgt_len[i] - 1] = EOS_ID

        return decoded




class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp)
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length











