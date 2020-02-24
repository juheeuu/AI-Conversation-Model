from torch.utils.data import Dataset, DataLoader
import pickle
import logging
import sentencepiece as spm


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

        assert len(conversation) == conversation_length == len(utterance_length)

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

class ConvPTBDataset(ConvDataset):
    def __init__(self, convs, utterances_length, vocab):
        """
        Dataset class for conversation
        Dataset class for conversation
        :param convs: A list of conversation that is represented as a list of utterances
        :param convs_length: A list of integer that indicates the number of utterances in each conversation
        :param utterances_length: A list of list whose element indicates the number of tokens in each utterance
        :param vocab: vocab class
        """
        self.convs = convs
        self.vocab = vocab
        self.len = len(convs)   # total number of conversations
        self.utterances_length = utterances_length

    def __getitem__(self, index):
        utterances = self.convs[index]
        target_utterance = utterances[-1]
        input_utterances = utterances[:-1]

        input_utterances_list = []
        for utter in input_utterances: 
            for i, tok in enumerate(utter): 
                if tok == '<eos>':
                    input_utterances_list += utter[:i+1]
                    input_utterances_list.append('<sep>')
        
        input_utterances_list.pop()
        input_utterances = input_utterances_list
        input_utterances = self.set_padding(input_utterances)

        target_utterance = ['<sos>'] + target_utterance + ['<eos>']
        target_utterance = self.set_padding(target_utterance)
            
        input_utterances_mask = [0 if tok == '<pad>' else 1 for tok in input_utterances]
        target_utterance_mask = [0 if tok == '<pad>' else 1 for tok in target_utterance]

        input_utterances = self.vocab.sent2id(input_utterances)
        target_utterance = self.vocab.sent2id(target_utterance)

        return input_utterances, input_utterances_mask, target_utterance, target_utterance_mask
    
    def set_padding(self, utterance, max_seq_len=512):
        if len(utterance) <= max_seq_len:
            utterance = utterance + ['<pad>' for _ in range(max_seq_len - len(utterance))]
        else:
            utterance.reverse()
            utterance = utterance[:max_seq_len]
            utterance.reverse()
        return utterance

class TransformerBasedConvDataset(Dataset):
    def __init__(self, convs, vocab, config):
        self.vocab = vocab 
        self.convs = convs 
        self.len = len(convs)
        self.max_seq_len = config.max_seq_len
        self.users = config.users 
    
    def __getitem__(self, index):

        conv = self.convs[index]

        inputs = []
        input_users = []

        for i, elem in enumerate(conv):
            if isinstance(elem, list):
                user_num = int(elem[0].replace('u', '').strip()) + 1
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


def get_loader(convs, vocab, convs_length=None, utterances_length=None, convs_users=None, batch_size=100, 
                shuffle=True, is_ptb_model=False, model=None, dataset=None, config=None):
    def collate_fn(data):
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)
        return zip(*data)

    if (model == "ZHENG" or model == "Transformer")  and (dataset == "cornell2" or dataset == "ubuntu"):
        dataset = TransformerBasedConvDataset(convs, vocab, config)
    elif (dataset == "cornell2" or dataset == "ubuntu") and model == "HRED":
        dataset = Cornell2HREDDataset(convs, vocab, config)
    elif convs_users is None:
        dataset = ConvDataset(convs, convs_length, utterances_length, vocab)
    elif is_ptb_model:
        dataset = ConvPTBDataset(convs, utterances_length, vocab)
    else:
        dataset = ConvUserDataset(convs, convs_users, convs_length, utterances_length, vocab)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader
