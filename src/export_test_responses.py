from config import get_config
from utils import Vocab, get_loader, load_pickle, PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, UNK_TOKEN, SEP_TOKEN
import solvers
import torch
from transformers import OpenAIGPTTokenizer

def main():
    config = get_config(mode='test')


    if config.data_name == "cornell":
        vocab = Vocab()
        vocab.load(config.word2id_path, config.id2word_path, ptb=(config.model == "PTB"))
        print(f'Vocabulary size: {vocab.vocab_size}')
        config.vocab_size = vocab.vocab_size

        if config.users:
            test_users = load_pickle(config.convs_users_path)
            config.user_size = max([x for xx in test_users for x in xx]) + 1
            print(f'User size: {config.user_size}')
        else:
            test_users = None

        data_loader = get_loader(convs=load_pickle(config.convs_path),
                                convs_length=load_pickle(config.conversations_length_path),
                                utterances_length=load_pickle(config.utterances_length_path),
                                vocab=vocab, batch_size=config.batch_size, shuffle=False, convs_users=test_users,
                                is_ptb_model=(config.model=="PTB"))

    elif config.data_name == "cornell2" or config.data_name == "ubuntu" or config.data_name == "twitter_s":
        vocab = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        special_tokens = {
            'pad_token': PAD_TOKEN,
            'bos_token': SOS_TOKEN,
            'eos_token': EOS_TOKEN,
            'sep_token': SEP_TOKEN,
        }
        vocab.add_special_tokens(special_tokens)
        config.vocab_size = len(vocab)
        config.vocab = vocab
        config.pad_id = vocab.pad_token_id
        config.eos_id = vocab.eos_token_id 
        config.sos_id = vocab.bos_token_id 

        data_loader = get_loader(convs=load_pickle(config.convs_path),
                                    vocab=vocab, 
                                    batch_size=config.batch_size,
                                    model=config.model,
                                    dataset=config.data_name,
                                    config=config,
                                    shuffle=False)
    else: 
        raise ValueError("{} Sorry... We don't support that data".format(config.data_name))

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    test_solver = model_solver(config, None, data_loader, vocab=vocab, is_train=False)

    test_solver.build()
    test_solver.export_samples(config.beam_size)


if __name__ == '__main__':
    main()
