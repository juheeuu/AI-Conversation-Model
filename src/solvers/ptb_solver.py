import numpy as np
import torch
import torch.nn as nn
from layers import masked_cross_entropy
from utils import to_var
import os
from tqdm import tqdm
from math import isnan
import codecs
import sys
from .solver import Solver

class SolverPTB(Solver):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SolverPTB, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)
    
    def train(self):
        epoch_loss_history = list()
        min_validation_loss = sys.float_info.max

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i 
            batch_loss_history = list()
            self.model.train()
            n_total_words = 0
            for batch_i, (input_utterances,
                          input_utterances_mask,
                          target_utterance,
                          target_utterance_mask,
                          utterance_length) in enumerate(tqdm(self.train_data_loader, ncols=80)):

                target_utterance_length = [l for len_list in utterance_length for l in len_list[1:]]
                
                input_utterances = to_var(torch.LongTensor(input_utterances))
                input_utterances_mask = to_var(torch.LongTensor(input_utterances_mask))
                target_utterance = to_var(torch.LongTensor(target_utterance))
                target_utterance_mask = to_var(torch.LongTensor(target_utterance_mask))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))

                self.optimizer.zero_grad()
                utterance_logits = self.model(
                    input_utterances, 
                    input_utterances_mask, 
                    target_utterance, 
                    target_utterance_mask
                )

                batch_loss, n_words = masked_cross_entropy(utterance_logits, target_utterance, target_utterance_length)

                assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item())
                n_total_words += n_words.item()

                if batch_i % self.config.print_every == 0:
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item()/ n_words.item():.3f}')

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            epoch_loss = np.sum(batch_loss_history) / n_total_words
            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print(f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}')
        
            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

            if min_validation_loss > self.validation_loss:
                min_validation_loss = self.validation_loss
            else:
                patience_cnt -= 1
                self.save_model(epoch_i)

            if patience_cnt < 0:
                print(f'\nEarly stop at {epoch_i}')
                self.save_model(epoch_i)
                return epoch_loss_history

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def evaluate(self):
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0

        for batch_i, (input_utterances, target_utterance, utterances_length) in \
            enumerate(tqdm(self.train_data_loader, ncols=80)):

            target_utterance_length = [l for len_list in utterances_length for l in len_list[1:]]
                
            with torch.no_grad():
                input_utterances = to_var(torch.LongTensor(input_utterances))
                target_utterance = to_var(torch.LongTensor(target_utterance))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))

            self.optimizer.zero_grad()
            utterance_logits = self.model(input_utterances, target_utterance_length, target_utterance_length)

            batch_loss, n_words = masked_cross_entropy(utterance_logits, target_utterances, target_utterance_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()
        
        epoch_loss = np.sum(batch_loss_history) / n_total_words
        print(f'Validation loss: {epoch_loss:.3f}\n')

        return epoch_loss

    def test(self):
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0
        for batch_i, (input_utterances, target_utterance, utterances_length) in \
             enumerate(tqdm(self.train_data_loader, ncols=80)):
            
            target_utterance_length = [l for len_list in utterances_length for l in len_list[1:]]
                
            with torch.no_grad():
                input_utterances = to_var(torch.LongTensor(input_utterances))
                target_utterance = to_var(torch.LongTensor(target_utterance))
                target_utterance_length = to_var(torch.LongTensor(target_utterance_length))

            self.optimizer.zero_grad()
            utterances_logits = self.model(input_utterances, target_utterance, target_utterance_length)

            batch_loss, n_words = masked_cross_entropy(utterances_logits, target_utterance, target_utterance_length)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item())
            n_total_words += n_words.item()

        epoch_loss = np.sum(batch_loss_history) / n_total_words
        
        print(f'Number of words: {n_total_words}')
        print(f'Bits per word: {epoch_loss:.3f}')
        word_perplexity = np.exp(epoch_loss)
        print(f'Word perplexity : {word_perplexity:.3f}\n')

        return word_perplexity

    def export_samples(self, beam_size=5):
        pass

