from torch.utils.data import Dataset, DataLoader
import pickle
import logging
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
import torch 

class ConvDataset(Dataset):
    def __init__(self, convs, convs_length, utterances_length, vocab):
        """
        Dataset class for conversation
        :param convs: A list of conversation that is represented as a list of utterances
        :param convs_length: A list of integer that indicates the number of utterances in each conversation
        :param utterances_length: A list of list whose element indicates the number of tokens in each utterance
        :param vocab: vocab class
        """
        self.convs = convs
        self.vocab = vocab
        self.convs_length = convs_length
        self.utterances_length = utterances_length
        self.len = len(convs)   # total number of conversations

    def __getitem__(self, index):
        """
        Extract one conversation
        :param index: index of the conversation
        :return: utterances, conversation_length, utterance_length
        """
        utterances = self.convs[index]
        conversation_length = self.convs_length[index]
        utterance_length = self.utterances_length[index]
        utterances = self.sent2id(utterances)

        return utterances, conversation_length, utterance_length

    def __len__(self):
        return self.len

    def sent2id(self, utterances):
        return [self.vocab.sent2id(utter) for utter in utterances]

class Cornell2HREDDataset(Dataset):
    def __init__(self, convs, vocab, config):
        self.vocab = vocab 
        self.convs = convs 
        self.len = len(convs)
        self.max_seq_len = config.max_seq_len

    def __getitem__(self, index):
        conv = self.convs[index]
        if isinstance(conv[0], list):
            conv = [info[1] for info in conv]
        conv = [ self.vocab.encode(utter, max_length=self.max_seq_len) + [self.vocab.eos_token_id] \
                for utter in conv]

        if len(conv) > 10:
            conv = conv[:10]

        conversation_length = len(conv)
        utterance_length = [len(utter) if len(utter) < self.max_seq_len else self.max_seq_len for utter in conv] 
        conversation = [self._set_padding(utter) for utter in conv]

        return conversation, conversation_length, utterance_length
    
    def __len__(self):
        return self.len

    def _set_padding(self, text):
        if len(text) <= self.max_seq_len:
            return text + [self.vocab.pad_token_id for _ in range(self.max_seq_len - len(text))]
        else:
            return text[len(text) - self.max_seq_len:]
    
class ConvUserDataset(ConvDataset):
    def __init__(self, convs, convs_users, convs_length, utterances_length, vocab):
        """
        Dataset class for conversation
        :param convs: A list of conversation that is represented as a list of utterances
        :param convs_users: A list of list whose element indicates the user index of each utterance
        :param convs_length: A list of integer that indicates the number of utterances in each conversation
        :param utterances_length: A list of list whose element indicates the number of tokens in each utterance
        :param vocab: vocab class
        """
        self.convs = convs
        self.vocab = vocab
        self.convs_length = convs_length
        self.utterances_length = utterances_length
        self.len = len(convs)   # total number of conversations
        self.convs_users = convs_users

    def __getitem__(self, index):
        """
        Extract one conversation
        :param index: index of the conversation
        :return: utterances, conversation_length, utterance_length
        """
        utterances = self.convs[index]
        conversation_length = self.convs_length[index]
        utterance_length = self.utterances_length[index]
        conversation_users = self.convs_users[index]

        utterances = self.sent2id(utterances)

        return utterances, conversation_length, utterance_length, conversation_users

class TransformerBasedConvDataset(Dataset):
    def __init__(self, convs, vocab, config):
        self.vocab = vocab 
        self.convs = convs 
        self.len = len(convs)
        self.max_seq_len = config.max_seq_len
        self.users = config.users 
        self.n_context = config.n_context
    
    def __getitem__(self, index):

        conv = self.convs[index]

        if self.n_context != 0: 
            conv = conv[:self.n_context+1]

        inputs = []
        input_users = []

        for i, elem in enumerate(conv):
            if isinstance(elem, list):
                user_num = int(elem[0].replace('u', '').strip()) + 1 if isinstance(elem[0], str) else elem[0] + 1
                utter = elem[1]
            else:
                user_num = None 
                utter = elem 

            if i < len(conv) - 1: 
                eos_tokens = [self.vocab.eos_token_id] if i == len(conv) - 2 else [self.vocab.eos_token_id, self.vocab.sep_token_id]
                encoded = self.vocab.encode(utter.strip(), max_length=self.max_seq_len) + eos_tokens
                inputs += encoded
                if user_num:
                    input_users += [user_num] * len(encoded)
            else: 
                target = [self.vocab.bos_token_id] + self.vocab.encode(utter, max_length=self.max_seq_len-2) + [self.vocab.eos_token_id]
                target_users = [user_num] * len(target) if user_num else None 

        input_utter, input_mask = self._setting(inputs)
        target_utter, target_mask = self._setting(target)

        if input_users and target_users and self.users:
            input_users = self._setting(input_users, pad_index=0, return_mask=False)
            target_users = self._setting(target_users, 0, False)
            assert len(input_users) == len(target_users) == self.max_seq_len
        else:
            input_users = None 
            target_users = None

        assert len(input_utter) == len(input_mask) == len(target_utter) == len(target_mask) == self.max_seq_len

        return input_utter, input_mask, target_utter, target_mask, input_users, target_users

    def __len__(self):
        return self.len 
    
    def _setting(self, text, pad_index=None, return_mask=True):
        if pad_index is None:
            pad_index = self.vocab.pad_token_id
            
        if len(text) <= self.max_seq_len:
            text = text + [pad_index for _ in range(self.max_seq_len - len(text))]
        else:
            text = text[len(text) - self.max_seq_len:]
            assert len(text) == self.max_seq_len

        if return_mask:
            text_mask = [ 0 if tok == self.vocab.pad_token_id else 1 for tok in text ]
            return text, text_mask
        
        return text 

class DialoGPTFeature(object):
    def __init__(self, input_ids, position_ids, token_type_ids, lm_labels, user_ids):
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.lm_labels = lm_labels
        self.user_ids = user_ids 

class DialoGPTDataset(Dataset):
    def __init__(self, convs, vocab, config):
        self.vocab = vocab 
        self.convs = convs 
        self.len = len(convs)
        self.max_seq_len = config.max_seq_len
        self.config = config
    
    def __len__(self):
        return self.len 
    
    def __getitem__(self, index):
        # conversation = user1 : [hi] / user2 : [how] [are] [u?] / user3: [fine] [thank] [you.]
        # input_ids =      [hi] [eos] [how] [are] [u?] [eos] [fine] [thank] [you]
        # position_ids =   [0]  [1]   [2]   [3]   [4]  [5]   [6]    [7]     [8]
        # token_type_ids = [0]  [1]   [1]   [1]   [1]  [2]   [2]    [2]     [2]
        # lm_labels      = [-1] [how] [are] [u?] [eos] [fine] [thank] [you] [eos]  

        eos_id = self.vocab.encoder['<|endoftext|>']

        conv = self.convs[index]
        max_seq_len = self.max_seq_len

        token_type_ids = []
        lm_labels = []

        # processed for max sequence length
        len_ = 0
        conv_ids = []
        for elem in conv:
            utter_id = self.vocab.encode(elem[1], max_length=(self.max_seq_len-2)) if isinstance(elem, list) \
                                                                                    or isinstance(elem, tuple) else self.vocab.encode(elem)
            len_ += len(utter_id)
            if len_ > self.max_seq_len - len(conv) - 2:
                if len(conv_ids) == 1:
                    utter_id = utter_id[:self.max_seq_len - len(conv) - 2 - len(conv_ids[0])]
                    conv_ids.append(utter_id)
                break
            conv_ids.append(utter_id) 
        
        assert len(conv_ids) >= 2

        if self.config.reversed:
            conv_ids = list(reversed(conv_ids))

        input_ids = [i for s in conv_ids for i in s+[eos_id]][:-1]
        user_ids = []
        for i, conv_id in enumerate(conv_ids): 
            user_id = int(elem[0].replace('u', '').strip()) + 1 if isinstance(conv[i][0], str) else conv[i][0]
            if i == 0: 
                lm_labels += [-1] * len(conv_id)
                token_type_ids += [0] * len(conv_id)

                user_ids += [user_id] * len(conv_id)
            else:
                lm_labels += conv_id + [eos_id]
                token_type_ids += [i] * (len(conv_id) + 1)
                user_ids += [user_id] * (len(conv_id) + 1)
        position_ids = list(range(len(input_ids)))
        assert (len(input_ids) == len(position_ids) == len(token_type_ids) == len(lm_labels) == len(user_ids))

        return DialoGPTFeature(input_ids, position_ids, token_type_ids, lm_labels, user_ids)

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        user_ids = pad_sequence([torch.tensor(f.user_ids, dtype=torch.long)
                                for f in features],
                                batch_first=True, padding_value=0)
        return (input_ids, position_ids, token_type_ids, labels, user_ids)



def get_loader(convs, vocab, convs_length=None, utterances_length=None, convs_users=None, batch_size=100, 
                shuffle=True, model=None, dataset=None, config=None):
    def collate_fn(data):
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)
        return zip(*data)
    
    if (model == "DialoGPT"):
        dataset = DialoGPTDataset(convs, vocab, config)
        collate_fn = DialoGPTDataset.collate
    elif (model == "ZHENG" or model == "Transformer")  and (dataset == "cornell2" or dataset == "ubuntu" or dataset=="twitter_s"):
        dataset = TransformerBasedConvDataset(convs, vocab, config)
    elif model == "HRED" and (dataset == "cornell2" or dataset == "ubuntu" or dataset == "twitter_s"):
        dataset = Cornell2HREDDataset(convs, vocab, config)
    elif convs_users is None:
        dataset = ConvDataset(convs, convs_length, utterances_length, vocab)
    else:
        dataset = ConvUserDataset(convs, convs_users, convs_length, utterances_length, vocab)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=True)

    return data_loader
