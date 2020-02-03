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
        """
        Extract one conversation
        :param index: index of the conversation
        :return: utterances, conversation_length, utterance_length
        """

        """
        it need <sep> token for each conversation 
        """
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

        ground_truth_target_utterance = target_utterance
        ground_truth_target_utterance = self.set_padding(ground_truth_target_utterance)
        
        target_utterance = ['<sos>'] + target_utterance
        target_utterance = self.set_padding(target_utterance)
            
        input_utterances_mask = [tok == '<pad>' for tok in input_utterances]
        target_utterance_mask = [tok == '<pad>' for tok in target_utterance]

        input_utterances = self.vocab.sent2id(input_utterances)
        target_utterance = self.vocab.sent2id(target_utterance)
        ground_truth_target_utterance = self.vocab.sent2id(ground_truth_target_utterance)

        return input_utterances, input_utterances_mask, target_utterance, target_utterance_mask, ground_truth_target_utterance
    
    def set_padding(self, utterance, max_seq_len=512):
        if len(utterance) <= max_seq_len:
            utterance = utterance + ['<pad>' for _ in range(max_seq_len - len(utterance))]
        else:
            utterance.reverse()
            utterance = utterance[:max_seq_len]
            utterance.reverse()
        return utterance



def get_loader(convs, convs_length, utterances_length, vocab, convs_users=None, batch_size=100, shuffle=True, is_ptb_model=False):
    def collate_fn(data):
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)
        return zip(*data)

    if convs_users is None and not is_ptb_model:
        dataset = ConvDataset(convs, convs_length, utterances_length, vocab)
    elif is_ptb_model:
        dataset = ConvPTBDataset(convs, utterances_length, vocab)
    else:
        dataset = ConvUserDataset(convs, convs_users, convs_length, utterances_length, vocab)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader


class LMDataSet(Dataset):
    def __init__(self, tokenizer, file_path: str, cached_path=None):

        if os.path.exists(cached_path):
            logger.info("Loading features from cached file %s", cached_path)
            with open(cached_path, "rb") as fb: 
                self.examples = pickle.load(fb)
        else: 
            logger.info("Creating features from dataset file at %s", file_path)

            self.examples = []

            with open(file_path, encoding="utf-8") as f: 
                text = f.readlines()
                for line in text: 
                    pass

            




def get_lm_loader(data_dir, spm_model):

    assert os.path.exists(data_dir)
    assert os.path.exists(spm_model)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(spm_model)

    file_path = data_dir
    cached_path = data_dir

    return LMDataSet(tokenizer, file_path, cached_path)
