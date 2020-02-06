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

        inputs = []
        for utter in input_utterances: 
            inputs += self.vocab.encode(utter)
            inputs += [self.vocab.eos_token_id, self.vocab.sep_token_id]
        inputs.pop()
        inputs = self.set_padding(inputs)

        gt_target = self.vocab.encode(target_utterance) + [self.vocab.eos_token_id]
        gt_target = self.set_padding(gt_target)

        target = [self.vocab.bos_token_id] + gt_target[:-1]

        input_mask = [ 0 if tok == self.vocab.pad_token_id else 1 for tok in inputs]
        target_mask = [ 0 if tok == self.vocab.pad_token_id else 1 for tok in target]
        gt_target_mask = [ 0 if tok == self.vocab.pad_token_id else 1 for tok in gt_target ]

        return inputs, input_mask, target, target_mask, gt_target, gt_target_mask
    
    def set_padding(self, utterance, max_seq_len=512):
        if len(utterance) <= max_seq_len:
            utterance = utterance + [ self.vocab.pad_token_id for _ in range(max_seq_len - len(utterance))]
        else:
            utterance.reverse()
            utterance = utterance[:max_seq_len]
            utterance.reverse()
        return utterance

    def __len__(self):
        return len(self.convs)




def get_loader(convs, convs_length=None, utterances_length=None, vocab=None, convs_users=None, batch_size=100, shuffle=True, is_ptb_model=False):
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
    def __init__(self, tokenizer, file_path):

        self.tokenizer = tokenizer

        cached_path = file_path + '.cached'

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
                    line = line.strip()
                    line = self.set_padding(line)
                    self.examples.append(line)

            logger.info("Saving fatures into cached file %s", cached_path)
            with open(cached_path, "wb") as f: 
                pickle.dump(self.examples, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def set_padding(self, sentence, max_seq_len=512): 
        """
        Parameters
        sentence: str of sentence
        max_seq_len: int 
        """

        sentence = self.tokenizer.EncodeAsPieces(sentence)
        if len(utterance) <= max_seq_len:
            sentence = sentence + ['<pad>' for _ in range(max_seq_len - len(sentence))]
        else:
            sentence.reverse()
            sentence = sentence[:max_seq_len]
            sentence.reverse()
        return [self.tokenizer.PieceToId(tok) for tok in sentence]

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)




def get_lm_loader(data_path, spm_model, batch_size=100, shuffle=True):

    assert os.path.exists(data_path)
    assert os.path.exists(spm_model)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(spm_model)

    return LMDataSet(tokenizer, data_path)
