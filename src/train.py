from utils import get_loader
from config import get_config
from utils import Vocab
import os
import solvers
from utils import load_pickle, PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, UNK_TOKEN, SEP_TOKEN
import torch 
import sentencepiece as spm
from transformers import OpenAIGPTTokenizer
import os

if __name__ == '__main__':
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    if config.data_name == "cornell":
        vocab = Vocab()
        vocab.load(config.word2id_path, config.id2word_path, ptb=(config.model == "PTB"))
        config.vocab_size = vocab.vocab_size
        config.pad_id = vocab.pad_id

        print(f'Vocabulary size: {vocab.vocab_size}')

        if config.users:
            train_users = load_pickle(config.convs_users_path)
            config.user_size = max([x for xx in train_users for x in xx]) + 1
            print(f'User size: {config.user_size}')
            eval_users = load_pickle(val_config.convs_users_path)
        else:
            train_users = None
            eval_users = None
        
            
        train_data_loader = get_loader(convs=load_pickle(config.convs_path),
                                    convs_length=load_pickle(config.conversations_length_path),
                                    utterances_length=load_pickle(config.utterances_length_path),
                                    vocab=vocab, convs_users=train_users,
                                    batch_size=config.batch_size,
                                    is_ptb_model=(config.model=="ZHENG") or (config.model=="Transformer"))

        eval_data_loader = get_loader(convs=load_pickle(val_config.convs_path),
                                    convs_length=load_pickle(val_config.conversations_length_path),
                                    utterances_length=load_pickle(val_config.utterances_length_path),
                                    vocab=vocab, shuffle=False, convs_users=eval_users,
                                    batch_size=val_config.eval_batch_size,
                                    is_ptb_model=(val_config.model=="ZHENG") or (val_config.model=="Transformer"))
    
    elif config.data_name == "cornell2" or "ubuntu":
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

        train_data_loader = get_loader(convs=load_pickle(config.convs_path),
                                        vocab=vocab, 
                                        batch_size=config.batch_size,
                                        model=config.model,
                                        dataset=config.data_name,
                                        config=config)
        
        eval_data_loader = get_loader(convs=load_pickle(val_config.convs_path),
                                        vocab=vocab,
                                        batch_size=val_config.batch_size,
                                        model=val_config.model,
                                        dataset=config.data_name,
                                        config=config)

    else: 
        raise ValueError("{} Sorry... We don't support that data".format(config.data_name))

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    solver = model_solver(config, train_data_loader, eval_data_loader, vocab=vocab, is_train=True)

    solver.build()
    solver.train()
