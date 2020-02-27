import numpy as np
import torch
import torch.nn as nn
from layers import masked_cross_entropy
from utils import to_var, PAD_ID, get_linear_schedule_with_warmup, EOS_ID, SOS_ID
import os
from tqdm import tqdm
from math import isnan
import codecs
import sys
from .solver import Solver
import torch.nn.functional as F
from subprocess import call

class SolverZHENG(Solver):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SolverZHENG, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)
    
    def train(self):
        epoch_loss_history = list()
        min_validation_loss = sys.float_info.max
        patience_cnt = self.config.patience

        self.config.n_gpu = torch.cuda.device_count()

        t_total = len(self.train_data_loader) * self.config.n_epoch
        cur_step = 0

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=t_total
        )

        if self.config.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model).to(self.config.device)
        else:
            self.model = self.model.to(self.config.device)

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i 
            batch_loss_history = list()
            self.model.train()

            epoch_lm_loss = 0.0 
            epoch_conv_loss = 0.0 
            epoch_batch_loss = 0.0 

            for batch_i, (input_utterances,
                          input_utterances_mask,
                          target_utterance,
                          target_utterance_mask,
                          input_user_ids,
                          target_user_ids) in enumerate(tqdm(self.train_data_loader, ncols=80)):
    
                input_utterances = torch.LongTensor(input_utterances).to(self.config.device)
                input_utterances_mask = torch.LongTensor(input_utterances_mask).to(self.config.device)
                target_utterance = torch.LongTensor(target_utterance).to(self.config.device)
                target_utterance_mask = torch.LongTensor(target_utterance_mask).to(self.config.device)

                user_available = input_user_ids[0] is not None 

                if user_available:
                    input_user_ids = torch.LongTensor(input_user_ids).to(self.config.device)
                    target_user_ids = torch.LongTensor(target_user_ids).to(self.config.device)

                self.optimizer.zero_grad()
                self.model.zero_grad()

                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_id)

                target, gt_target = target_utterance[..., :-1].contiguous(), target_utterance[..., 1:].contiguous()
                target_mask = target_utterance_mask[..., :-1].contiguous()
                
                if user_available:
                    target_user_ids = target_user_ids[..., :-1].contiguous()
                else:
                    input_user_ids = None 
                    target_user_ids = None 

                lm_output, conv_output = self.model(target, target_mask,
                                                    input_utterances, input_utterances_mask,
                                                    target_user_ids, input_user_ids)
                
                # 1. Calculate Language Model Loss 
                outputs, labels = lm_output[..., :-1, :].contiguous(), input_utterances[..., 1:].contiguous()
                lm_loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                # 2. Calculate Conv Loss 
                conv_loss = loss_fn(conv_output.view(-1, conv_output.size(-1)), gt_target.view(-1))

                # 3. Total Loss 
                batch_loss = lm_loss * 0.2 + conv_loss

                assert not isnan(batch_loss.item())

                if self.config.n_gpu > 1: 
                    batch_loss = batch_loss.mean()

                epoch_lm_loss = (batch_i * epoch_lm_loss + lm_loss.item()) / (batch_i + 1)
                epoch_conv_loss = (batch_i * epoch_conv_loss + conv_loss.item()) / (batch_i + 1)
                epoch_batch_loss = (batch_i * epoch_batch_loss + batch_loss.item()) / (batch_i + 1)

                if batch_i % self.config.print_every == 0:
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item():.3f}')
                    self.writer.add_scalar('Train/lm_loss', lm_loss.item(), cur_step)
                    self.writer.add_scalar('Train/conv_loss', conv_loss.item(), cur_step)
                    self.writer.add_scalar('Train/loss', batch_loss.item(), cur_step)
                    self.writer.add_scalar('Train/learning_rate', self.scheduler.get_lr()[0], cur_step)

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()
                self.scheduler.step()
                cur_step += 1

            epoch_loss_history.append(epoch_batch_loss)
            self.epoch_loss = epoch_batch_loss

            print(f'Epoch {epoch_i+1} loss average: {epoch_batch_loss:.3f}')

            print('\n<Validation>...')
            val_loss, val_lm_loss, val_conv_loss = self.evaluate()
            self.validation_loss = val_loss

            if epoch_i % self.config.plot_every_epoch == 0:
                self.writer.add_scalar('Val/lm_loss', val_lm_loss, epoch_i + 1)
                self.writer.add_scalar('Val/conv_loss', val_conv_loss, epoch_i + 1)
                self.writer.add_scalar('Val/loss', val_loss, epoch_i + 1)

            self.save_model(epoch_i)

            if min_validation_loss > self.validation_loss:
                min_validation_loss = self.validation_loss
            else:
                patience_cnt -= 1

            if patience_cnt < 0:
                print(f'\nEarly stop at {epoch_i}')
                return epoch_loss_history

        self.save_model(self.config.n_epoch)

        return epoch_loss_history

    def evaluate(self):
        self.model.eval()
        epoch_batch_loss = 0.0
        epoch_lm_loss = 0.0 
        epoch_conv_loss = 0.0

        for batch_i, (input_utterances,
                      input_utterances_mask,
                      target_utterance,
                      target_utterance_mask,
                      input_user_ids,
                      target_user_ids) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
                
            with torch.no_grad():
                input_utterances = torch.LongTensor(input_utterances).to(self.config.device)
                input_utterances_mask = torch.LongTensor(input_utterances_mask).to(self.config.device)
                target_utterance = torch.LongTensor(target_utterance).to(self.config.device)
                target_utterance_mask = torch.LongTensor(target_utterance_mask).to(self.config.device)

                user_available = input_user_ids[0] is not None 

                if user_available:
                    input_user_ids = torch.LongTensor(input_user_ids).to(self.config.device)
                    target_user_ids = torch.LongTensor(target_user_ids).to(self.config.device)

            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_id)

            target, gt_target = target_utterance[..., :-1].contiguous(), target_utterance[..., 1:].contiguous()
            target_mask = target_utterance_mask[..., :-1].contiguous()

            if user_available:
                target_user_ids = target_user_ids[..., :-1].contiguous()
            else:
                input_user_ids = None 
                target_user_ids = None 

            lm_output, conv_output = self.model(target, target_mask, 
                                                input_utterances, input_utterances_mask, 
                                                target_user_ids, input_user_ids)
            
            # 1. Calculate Language Model Loss 
            outputs, labels = lm_output[..., :-1, :].contiguous(), input_utterances[..., 1:].contiguous()
            lm_loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # 2. Calculate Conv Loss 
            conv_loss = loss_fn(conv_output.view(-1, outputs.size(-1)), gt_target.view(-1))

            # 3. Total Loss 
            batch_loss = lm_loss * 0.2 + conv_loss

            if self.config.n_gpu > 1:
                batch_loss = batch_loss.mean()

            assert not isnan(batch_loss.item())
        
            epoch_batch_loss = (batch_i * epoch_batch_loss + batch_loss.item()) / (batch_i + 1)
            epoch_lm_loss = (batch_i * epoch_lm_loss + lm_loss.item()) / (batch_i + 1)
            epoch_conv_loss = (batch_i * epoch_conv_loss + conv_loss.item()) / (batch_i + 1)
            
        print(f'Validation loss: {epoch_batch_loss:.3f}\n')

        return epoch_batch_loss, epoch_lm_loss, epoch_conv_loss

    
    def export_samples(self, beam_size, file_write=True):
        self.model.eval()
        n_sample_step = self.config.n_sample_step
        context_history = list()
        sample_history = list()
        ground_truth_history = list()
        generated_history = list()
        input_history = list()

        for batch_i, (input_utterances,
                      input_utterances_mask,
                      target_utterance,
                      _,
                      input_user_ids,
                      target_user_ids) in enumerate(tqdm(self.eval_data_loader, ncols=80)):

            context_history.append(input_utterances)
            with torch.no_grad():
                input_utterances = torch.LongTensor(input_utterances).to(self.config.device)
                input_utterances_mask = torch.LongTensor(input_utterances_mask).to(self.config.device)
                
                user_available = input_user_ids[0] is not None 
                
                if user_available:
                    input_user_ids = torch.LongTensor(input_user_ids).to(self.config.device)
                    target_user_ids = torch.LongTensor(target_user_ids).to(self.config.device)
                else:
                    input_user_ids = None 
                    target_user_ids = None 

            max_seq_len = self.model.config.max_seq_len 

            if beam_size == 1:
                # do Greedy Decoding 
                enc_hidden = self.model.encode(input_utterances, input_utterances_mask, input_user_ids)
                dec_input = torch.LongTensor([[self.config.vocab.bos_token_id]]).to(self.config.device)

                for i in range(max_seq_len):
                    dec_id = target_user_ids[...,:i+1] if user_available else None
                    y_pred = self.model.decode(dec_input, None, enc_hidden, input_utterances_mask, dec_id)
                    y_pred_ids = y_pred.max(dim=-1)[1]

                    new_word = y_pred_ids.tolist()[0][-1]

                    if new_word == self.config.vocab.eos_token_id or i == max_seq_len - 1:
                        break

                    dec_input = torch.cat((dec_input, torch.LongTensor([[new_word]]).to(self.config.device)), dim=-1)

                labels = y_pred_ids.tolist()
            else: 
                # Beam Decoding 
                labels = self.model.beam_generate(input_utterances, input_utterances_mask, input_user_ids, 
                                    target_user_ids, self.config.vocab.bos_token_id,
                                    self.config.vocab.pad_token_id, self.config.vocab.eos_token_id).tolist()
        
                
            input_utterances = input_utterances.tolist()
            ground_truthes = list(target_utterance)

            for label, input_utter, ground_truth in zip(labels, input_utterances, ground_truthes):
                label = self.vocab.convert_ids_to_tokens(label)
                label = self.vocab.convert_tokens_to_string(label)
                label = label.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").strip()
                generated_history.append(label)

                input_utter = self.vocab.convert_ids_to_tokens(input_utter)
                input_utter = self.vocab.convert_tokens_to_string(input_utter)
                input_utter = input_utter.replace("<pad>", "").strip()
                input_history.append(input_utter)

                ground_truth = self.vocab.convert_ids_to_tokens(ground_truth)
                ground_truth = self.vocab.convert_tokens_to_string(ground_truth)
                ground_truth = ground_truth.replace("<sos>", "").replace("<eos>", "").replace("<pad>", "").strip()
                ground_truth_history.append(ground_truth)

        if file_write:
            target_file_name = 'responses_{}_{}_{}.txt'.format(self.config.mode, beam_size, self.epoch_i)
            print("Writing candidates into file {}".format(target_file_name))
            conv_idx = 0 
            with codecs.open(os.path.join(self.config.save_path, target_file_name), 'w', "utf-8") as output_f:
                for input_utter, generated, ground_truth in tqdm(zip(input_history, generated_history, ground_truth_history)):
                    print("Conversation Context {}".format(conv_idx), file=output_f)
                    print(input_utter, file=output_f)
                    print(generated, file=output_f)
                    print(ground_truth, file=output_f)
                    conv_idx += 1
            return conv_idx
        
        else: 
            return input_history, generated_history
