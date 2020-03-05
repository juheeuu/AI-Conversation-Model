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

class SolverDialoGPT(Solver):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SolverDialoGPT, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)

    def train(self):
        epoch_loss_history = list()
        min_validation_loss = sys.float_info.max
        patience_cnt = self.config.patience

        self.config.n_gpu = torch.cuda.device_count()

        t_total = len(self.train_data_loader) * self.config.n_epoch
        cur_step = 0

        no_decay = ['bias', 'ln']
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
            self.model.train()

            epoch_loss = 0.0

            for batch_i, batch in enumerate(tqdm(self.train_data_loader, ncols=80)):

                self.optimizer.zero_grad()
                self.model.zero_grad()

                batch = tuple(t.to(self.config.device) for t in batch)
                input_ids, position_ids, token_ids, label_ids = batch
                inputs = {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'token_type_ids': token_ids, 
                    'lm_labels': label_ids
                }

                outputs = self.model(**inputs)

                loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
                batch_loss = loss_fn(outputs.view(-1, outputs.size(-1)),
                                label_ids.view(-1))
                
                assert not isnan(batch_loss.item())

                if self.config.n_gpu > 1: 
                    batch_loss = batch_loss.mean()

                epoch_loss = (batch_i * epoch_loss + batch_loss.item()) / (batch_i + 1)

                if batch_i % self.config.print_every == 0:
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item():.3f}')
                    self.writer.add_scalar('Train/loss', batch_loss.item(), cur_step)
                    self.writer.add_scalar('Train/learning_rate', self.scheduler.get_lr()[0], cur_step)

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()
                self.scheduler.step()
                cur_step += 1

            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print(f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}')

            print('\n<Validation>...')
            self.validation_loss = self.evaluate()

            if epoch_i % self.config.plot_every_epoch == 0:
                self.writer.add_scalar('Val/loss', self.validation_loss, epoch_i + 1)

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
        epoch_loss = 0.0

        for batch_i, batch in enumerate(tqdm(self.eval_data_loader, ncols=80)):
                
            with torch.no_grad():
                batch = tuple(t.to(self.config.device) for t in batch)
            
            input_ids, position_ids, token_ids, label_ids = batch
            inputs = {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'token_type_ids': token_ids, 
                'lm_labels': label_ids
            }

            outputs = self.model(**inputs)

            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            batch_loss = loss_fn(outputs.view(-1, outputs.size(-1)),
                                label_ids.view(-1))

            if self.config.n_gpu > 1:
                batch_loss = batch_loss.mean()

            epoch_loss = (batch_i * epoch_loss + batch_loss.item()) / (batch_i + 1)

            assert not isnan(batch_loss.item())
        
        print(f'Validation loss: {epoch_loss:.3f}\n')
        return epoch_loss

    
    def export_samples(self, beam_size, file_write=True):
        self.model.eval()
        context_history = list()
        sample_history = list()
        ground_truth_history = list()
        generated_history = list()
        input_history = list()

        for batch_i, batch in enumerate(tqdm(self.eval_data_loader, ncols=80)):
           
            with torch.no_grad():
                batch = tuple(t.to(self.config.device) for t in batch)
            
            input_ids, position_ids, token_ids, label_ids = batch

            gt_mask = label_ids != -1 
            gt_ids = input_ids[gt_mask][1:]

            input_mask = label_ids == -1 
            input_mask = torch.cat((torch.BoolTensor([[True]]).to(self.config.device), input_mask),dim=1)[...,:-1]
            input_ids = input_ids[input_mask]
            input_ids = input_ids.unsqueeze(0)

            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=self.config.max_seq_len,
                temperature=0.9,
                top_k=0,
                top_p=0.9,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=1,
                eos_token_ids=self.config.vocab.eos_token_id
            )

            output_sequences.squeeze_()
            output = output_sequences.tolist()
            output = self.vocab.decode(output, clean_up_tokenization_spaces=True)
            output = output.split(self.vocab.eos_token)
            generated_history.append(output[1])

            input_ids.squeeze_()
            input_ids = input_ids.tolist()
            inputs = self.vocab.decode(input_ids, clean_up_tokenization_spaces=True)
            input_history.append(inputs)

            gt_ids.squeeze_()
            gt_ids = gt_ids.tolist()
            gt = self.vocab.decode(gt_ids, clean_up_tokenization_spaces=True)
            gt = gt.split(self.vocab.eos_token)
            ground_truth_history.append(gt[0])

        target_file_name = 'responses_{}_{}_{}_{}.txt'.format(self.config.mode, self.config.n_context, beam_size, self.epoch_i)
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