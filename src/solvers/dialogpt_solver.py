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
from models import DialoGPT
import copy

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

                input_ids, position_ids, token_ids, label_ids, user_ids, user_mask = batch

                inputs = {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'token_type_ids': token_ids, 
                    'user_mask': user_mask
                }

                outputs = self.model(**inputs)

                loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

                if self.config.users and self.config.reversed:
                    lm_logits, user_outputs = outputs
                    user_ids = user_ids.view(-1)
                    user_ids_mask = user_ids != -1 
                    user_ids = user_ids[user_ids_mask]
                    user_loss = loss_fn(user_outputs.view(-1, user_outputs.size(-1)),
                        user_ids.view(-1))
                    output_loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)),
                                label_ids.view(-1))
                    batch_loss = user_loss + output_loss
                else:
                    batch_loss = loss_fn(outputs.view(-1, outputs.size(-1)),
                                label_ids.view(-1))
                    user_loss = None
                    output_loss = None
                
                assert not isnan(batch_loss.item())

                if self.config.n_gpu > 1: 
                    batch_loss = batch_loss.mean()

                epoch_loss = (batch_i * epoch_loss + batch_loss.item()) / (batch_i + 1)

                if batch_i % self.config.print_every == 0:
                    tqdm.write(f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {batch_loss.item():.3f}')
                    self.writer.add_scalar('Train/loss', batch_loss.item(), cur_step)
                    if self.config.users and self.config.reversed:
                        self.writer.add_scalar('Train/user_loss', user_loss.item(), cur_step)
                        self.writer.add_scalar('Train/lmloss', output_loss.item(), cur_step)
                    self.writer.add_scalar('Train/learning_rate', self.scheduler.get_lr()[0], cur_step)

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()
                self.scheduler.step()
                cur_step += 1
            
            if epoch_i == 0:
                if self.config.users and not self.config.reversed:
                    self.vocab.save_pretrained(self.config.save_path)

            epoch_loss_history.append(epoch_loss)
            self.epoch_loss = epoch_loss

            print(f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}')

            print('\n<Validation>...')
            val_loss, output_loss, user_loss = self.evaluate()
            self.validation_loss  = val_loss

            if epoch_i % self.config.plot_every_epoch == 0:
                self.writer.add_scalar('Val/loss', self.validation_loss, epoch_i + 1)
                if output_loss != None and user_loss != None:
                    self.writer.add_scalar('Val/lmloss', output_loss.item(), epoch_i + 1)
                    self.writer.add_scalar('Val/user_loss', user_loss.item(), epoch_i + 1)

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
            
            input_ids, position_ids, token_ids, label_ids, user_ids, user_mask = batch
            inputs = {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'token_type_ids': token_ids, 
                'user_mask': user_mask,
            }

            outputs = self.model(**inputs)
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

            if self.config.users and self.config.reversed:
                lm_logits, user_outputs = outputs
                user_ids = user_ids.view(-1)
                user_ids_mask = user_ids != -1 
                user_ids = user_ids[user_ids_mask]
                user_loss = loss_fn(user_outputs.view(-1, user_outputs.size(-1)),
                    user_ids.view(-1))
                output_loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)),
                            label_ids.view(-1))
                batch_loss = user_loss + output_loss
            else: 
                batch_loss = loss_fn(outputs.view(-1, outputs.size(-1)),
                                                    label_ids.view(-1))
                output_loss = None
                user_loss = None

            

            if self.config.n_gpu > 1:
                batch_loss = batch_loss.mean()

            epoch_loss = (batch_i * epoch_loss + batch_loss.item()) / (batch_i + 1)

            assert not isnan(batch_loss.item())
        
        print(f'Validation loss: {epoch_loss:.3f}\n')
        return epoch_loss, output_loss, user_loss

    
    def export_samples(self, beam_size, file_write=True):
        self.model.eval()
        context_history = list()
        sample_history = list()
        ground_truth_history = list()
        generated_history = list()
        input_history = list()

        if self.config.mmi:
            reversed_config = copy.deepcopy(self.config)
            reversed_config.pretrained_path = self.config.reversed_pretrained_path
            reversed_config.reversed = True
            reversed_config.original = False
            self.reversed_model = DialoGPT(reversed_config).cuda(1)

        for batch_i, batch in enumerate(tqdm(self.eval_data_loader, ncols=80)):
           
            with torch.no_grad():
                batch = tuple(t.to(self.config.device) for t in batch)
            
            input_ids, position_ids, token_ids, label_ids, user_ids, \
                user_masks = batch

            gt_mask = label_ids != -1 
            gt_ids = input_ids[gt_mask][1:]

            input_mask = label_ids == -1 
                
            input_mask = torch.cat((torch.BoolTensor([[True]]).to(self.config.device), input_mask),dim=1)[...,:-1]
            input_ids = input_ids[input_mask]
            input_ids = input_ids.unsqueeze(0)

            if self.config.mmi:
                num_return_sequences = 3
            else:
                num_return_sequences = 1

            output_sequences = self.model.gpt2.generate(
                input_ids=input_ids,
                max_length=self.config.max_seq_len-20,
                temperature=0.9,
                top_k=0,
                top_p=0.9,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.vocab.pad_token,
            )

            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

            if self.config.mmi:
                results = []
                if self.config.users:
                    user_mask = input_ids > 50256
                    user_ids = input_ids[user_mask]
                    user_ids = user_ids[1]

                    user_id = user_ids.tolist()
                    user_ids_str = self.vocab.decode([user_id])
                    user_ids = torch.LongTensor([int(user_ids_str[1:])]).cuda(1)

                    user_ids.unsqueeze(0)

                    input_ids_mask = input_ids <= 50256
                    input_ids = input_ids[input_ids_mask]

                for seq in output_sequences:
                    # reverse_input_seq 
                    if self.config.users:
                        seq = seq.unsqueeze(0).cuda(1)
                        original_seq = seq.unsqueeze(0).cuda(1)
                        seq_mask = seq <= 50256
                        seq = seq[seq_mask]
                        seq = seq[:self.config.max_seq_len-200]
                        input_ids = input_ids[:self.config.max_seq_len-200]

                        inputs = torch.cat((seq, input_ids), dim=-1).cuda(1)
                        mask = torch.full_like(seq, -1, dtype=torch.long).cuda(1)
                        labels = torch.cat((mask, input_ids), dim=-1).cuda(1)
                        user_mask = torch.LongTensor([[0] * (len(seq)-1) + [1] + [0] * len(input_ids)])

                        inputs = inputs
                        labels = torch.cat((mask, input_ids), dim=-1).cuda(1)

                        loss, user_output = self.reversed_model(inputs, lm_labels=labels, user_ids=user_ids,user_mask=user_mask)
                        user_loss = loss_fn(user_output.view(-1, user_output.size(-1)), user_ids.view(-1))
                        loss = loss + user_loss
                        results.append((original_seq, -loss.float()))
                    else:
                        original_seq=seq.unsqueeze(0).cuda(1)
                        seq = seq.unsqueeze(0).cuda(1)

                        input_ids_for_mmi = input_ids.tolist()[0]
                        size = len(input_ids_for_mmi) 
                        idx_list = [idx + 1 for idx, val in
                                    enumerate(input_ids_for_mmi) if val == 50256] 
                        res = [input_ids_for_mmi[i: j] for i, j in
                                zip([0] + idx_list, idx_list + 
                                ([size] if idx_list[-1] != size else []))] 
                        res.reverse()
                        input_ids_reversed = [item for sublist in res for item in sublist]
                        input_ids_reversed = torch.LongTensor(input_ids_reversed).unsqueeze(0).cuda(1)

                        inputs = torch.cat((seq, input_ids_reversed), dim=-1).cuda(1)
                        mask = torch.full_like(seq, -1, dtype=torch.long).cuda(1)
                        inputs = inputs[:,:self.config.max_seq_len]
                        labels = torch.cat((mask, input_ids_reversed), dim=-1).cuda(1)[:,:self.config.max_seq_len]
                        loss, *_ = self.reversed_model(inputs, lm_labels=labels)
                        results.append((original_seq.cpu(), -loss.float()))

                
                MMI_temperature = 0.5
                scores = torch.stack([x[1] for x in results], dim=0)
                winner = torch.multinomial(F.softmax(scores / MMI_temperature, dim=0), num_samples=1).item()

                output_sequences = results[winner][0]
            output_sequences.squeeze_()
            output = output_sequences.tolist()
            output = self.vocab.decode(output, clean_up_tokenization_spaces=True)

            output = output.split(self.vocab.eos_token)
            assert (len(output) >= 2)
            gen_history = output[self.config.n_context].replace("\n", " ")
            generated_history.append(gen_history)

            input_ids.squeeze_()
            input_ids = input_ids.tolist()
            inputs = self.vocab.decode(input_ids, clean_up_tokenization_spaces=True)
            inputs.replace("\n", " ")
            input_history.append(inputs)

            gt_ids.squeeze_()
            gt_ids = gt_ids.tolist()
            gt = self.vocab.decode(gt_ids, clean_up_tokenization_spaces=True)
            gt = gt.split(self.vocab.eos_token)
            ground_truth = gt[0].replace("\n", " ")
            ground_truth_history.append(ground_truth)

        target_file_name = 'responses_{}_{}_{}_{}.txt'.format(self.config.mode, self.config.n_context, beam_size, self.epoch_i)
        if self.config.mmi:
            target_file_name = target_file_name.replace('.txt', '_mmi.txt')
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
    
