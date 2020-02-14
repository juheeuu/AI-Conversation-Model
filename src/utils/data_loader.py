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

class Cornell2TransformerBasedDataset(Dataset):
    def __init__(self, convs, vocab, config):
        self.vocab = vocab 
        self.convs = convs 
        self.len = len(convs)
        self.max_seq_len = config.max_seq_len
    
    def __getitem__(self, index):
        conv = self.convs[index]
        input_utters = conv[:-1]
        target_utter = conv[-1]

        inputs = []
        for utter in input_utters:
            inputs += self.vocab.encode(utter, max_length=self.max_seq_len)
            inputs += [self.vocab.eos_token_id, self.vocab.sep_token_id]
        inputs = inputs[:-1]

        target_utter = [self.vocab.bos_token_id] + self.vocab.encode(target_utter, max_length=self.max_seq_len-2) + [self.vocab.eos_token_id]

        input_utter, input_mask = self._setting(inputs)
        target_utter, target_mask = self._setting(target_utter)

        return input_utter, input_mask, target_utter, target_mask

    def __len__(self):
        return self.len 
    
    def _setting(self, text):
        if len(text) <= self.max_seq_len:
            text = text + [self.vocab.pad_token_id for _ in range(self.max_seq_len - len(text))]
        else:
            text = text[len(text) - self.max_seq_len:]
            assert len(text) == self.max_seq_len

        text_mask = [ 0 if tok == self.vocab.pad_token_id else 1 for tok in text ]
        return text, text_mask


def get_loader(convs, vocab, convs_length=None, utterances_length=None, convs_users=None, batch_size=100, 
                shuffle=True, is_ptb_model=False, model=None, dataset=None, config=None):
    def collate_fn(data):
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)
        return zip(*data)

    if (model == "ZHENG" or model == "Transformer")  and dataset == "cornell2":
        dataset = Cornell2TransformerBasedDataset(convs, vocab, config)
    elif convs_users is None and not is_ptb_model:
        dataset = ConvDataset(convs, convs_length, utterances_length, vocab)
    elif is_ptb_model:
        dataset = ConvPTBDataset(convs, utterances_length, vocab)
    else:
        dataset = ConvUserDataset(convs, convs_users, convs_length, utterances_length, vocab)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader
