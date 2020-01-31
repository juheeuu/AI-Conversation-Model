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
import torch.nn.functional as F

class SolverPTB(Solver):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SolverPTB, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)
    
    def train(self):
        epoch_loss_history = list()
        min_validation_loss = sys.float_info.max
        patience_cnt = self.config.patience
        step = 0

        self.config.n_gpu = torch.cuda.device_count()
        self.model = self.model.to(self.config.device)

        # if self.config.n_gpu > 1:
        #     self.model = torch.nn.DataParallel(self.model).to(self.config.device)

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i 
            batch_loss_history = list()
            self.model.train()
            n_total_words = 0
            for batch_i, (input_utterances,
                          input_utterances_mask,
                          target_utterance,
                          target_utterance_mask) in enumerate(tqdm(self.train_data_loader, ncols=80)):
    
                input_utterances = torch.LongTensor(input_utterances).to(self.config.device)
                input_utterances_mask = torch.BoolTensor(input_utterances_mask).to(self.config.device)
                target_utterance = torch.LongTensor(target_utterance).to(self.config.device)
                target_utterance_mask = torch.BoolTensor(target_utterance_mask).to(self.config.device)

                self.optimizer.zero_grad()
                utterance_logits = self.model(
                    input_utterances, 
                    input_utterances_mask, 
                    target_utterance, 
                    target_utterance_mask
                )

                # masked cross entropy 
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
                active_loss = target_utterance_mask.view(-1) != True 
                utterance_logits = utterance_logits.view(-1, utterance_logits.size(2))
                target_utterance = target_utterance.view(-1)
                batch_loss = loss_fn(utterance_logits, target_utterance)
                target_utterance = target_utterance[active_loss]

                n_words = target_utterance.size(0)

                assert not isnan(batch_loss.item())
                batch_loss_history.append(batch_loss.item() * n_words)
                n_total_words += n_words

                step += 1

                if batch_i % self.config.print_every == 0:
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item():.3f}')
                    self.writer.update_loss(batch_loss.item(), step, 'train_loss')

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

        for batch_i, (input_utterances,
                      input_utterances_mask,
                      target_utterance,
                      target_utterance_mask) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
                
            with torch.no_grad():
                input_utterances = torch.LongTensor(input_utterances).to(self.config.device)
                input_utterances_mask = torch.BoolTensor(input_utterances_mask).to(self.config.device)
                target_utterance = torch.LongTensor(target_utterance).to(self.config.device)
                target_utterance_mask = torch.BoolTensor(target_utterance_mask).to(self.config.device)

            utterance_logits = self.model(input_utterances, input_utterances_mask, target_utterance, target_utterance_mask)

            # masked cross entropy 
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            active_loss = target_utterance_mask.view(-1) != True 
            utterance_logits = utterance_logits.view(-1, utterance_logits.size(2))
            target_utterance = target_utterance.view(-1)
            batch_loss = loss_fn(utterance_logits, target_utterance)
            target_utterance = target_utterance[active_loss]
            n_words = target_utterance.size(0)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item() * n_words)
            n_total_words += n_words
        
        epoch_loss = np.sum(batch_loss_history) / n_total_words
        print(f'Validation loss: {epoch_loss:.3f}\n')

        return epoch_loss

    def test(self):
        self.model.eval()
        batch_loss_history = []
        n_total_words = 0
        for batch_i, (input_utterances,
                      input_utterances_mask,
                      target_utterance,
                      target_utterance_mask) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
                
            with torch.no_grad():
                input_utterances = torch.LongTensor(input_utterances).to(self.config.device)
                input_utterances_mask = torch.BoolTensor(input_utterances_mask).to(self.config.device)
                target_utterance = torch.LongTensor(target_utterance).to(self.config.device)
                target_utterance_mask = torch.BoolTensor(target_utterance_mask).to(self.config.device)

            utterance_logits = self.model(input_utterances, input_utterances_mask, target_utterance, target_utterance_mask)

            # masked cross entropy 
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            active_loss = target_utterance_mask.view(-1) != True 
            utterance_logits = utterance_logits.view(-1, utterance_logits.size(2))
            target_utterance = target_utterance.view(-1)
            batch_loss = loss_fn(utterance_logits, target_utterance)
            target_utterance = target_utterance[active_loss]
            n_words = target_utterance.size(0)

            assert not isnan(batch_loss.item())
            batch_loss_history.append(batch_loss.item() * n_words)
            n_total_words += n_words

        epoch_loss = np.sum(batch_loss_history) / n_total_words
        
        print(f'Number of words: {n_total_words}')
        print(f'Bits per word: {epoch_loss:.3f}')
        word_perplexity = np.exp(epoch_loss)
        print(f'Word perplexity : {word_perplexity:.3f}\n')

        return word_perplexity

    def export_samples(self, beam_size=4):
        self.model.config.beam_size = beam_size
        self.model.eval()
        n_sample_step = self.config.n_sample_step
        context_history = list()
        sample_history = list()
        ground_truth_history = list()

        for batch_i, (input_utterances,
                      input_utterances_mask,
                      target_utterance,
                      target_utterance_mask) in enumerate(tqdm(self.eval_data_loader, ncols=80)):

            context_history.append(input_utterances)
            with torch.no_grad():
                input_utterances = torch.LongTensor(input_utterances).to(self.config.device)
                input_utterances_mask = torch.BoolTensor(input_utterances_mask).to(self.config.device)
            
            all_samples = self.model.generate(input_utterances, input_utterances_mask)

            all_samples = all_samples.data.cpu().numpy().tolist()
            sample_history.append(all_samples)
            ground_truth_history.append(target_utterance)

        
        target_file_name = 'responses_{}_{}_{}_{}.txt'.format(self.config.mode, n_sample_step,
                                                                 beam_size, self.epoch_i)
        print("Writing candidates into file {}".format(target_file_name))
        conv_idx = 0 
        with codecs.open(os.path.join(self.config.save_path, target_file_name), 'w', "utf-8") as output_f:
            for contexts, samples, ground_truths in tqdm(zip(context_history, sample_history, ground_truth_history),
                                                         total=len(context_history), ncols=80):
                for one_conv_contexts, one_conv_samples, one_conv_ground_truth in zip(contexts, samples, ground_truths):
                    print("Conversation Context {}".format(conv_idx), file=output_f)
                    print(self.vocab.decode(one_conv_contexts), file=output_f)
                    print(self.vocab.decode(one_conv_samples), file=output_f)
                    print(self.vocab.decode(one_conv_ground_truth), file=output_f)
                    conv_idx += 1

        return conv_idx

        





                



