import codecs
import numpy as np
import sys
from utils import bleu_compute, rouge_compute, rouge_names, to_var, embedding_compute, dist_compute
from utils import PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, UNK_TOKEN, SEP_TOKEN
from scipy.stats import sem
import tabulate
import torch.nn as nn 
import torch 
import pickle
from transformers import OpenAIGPTTokenizer, GPT2Tokenizer

def main():
    bleu_list = list()
    length_history = list()
    rouge_history = list()
    embedding_list = list()
    dist1_list = list()
    conv_idx_match = 0
    convs_top_answer = list()
    convs_ground_truth = list()
    num_answers = 1

    if dataset == "cornell2" or dataset == "ubuntu" or dataset == "twitter_s" or dataset=="bigbangtheory":
        if model_name == "DialoGPT":
            vocab = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            vocab = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            special_tokens = {
                'pad_token': PAD_TOKEN,
                'bos_token': SOS_TOKEN,
                'eos_token': EOS_TOKEN,
                'sep_token': SEP_TOKEN,
            }
            vocab.add_special_tokens(special_tokens)
        state_dict = torch.load(checkpoint_path)

        embedding_weight_name = None
        for key in state_dict.keys():
            if key.endswith("tok_embedding.weight"):
                embedding_weight_name = key 
                break 
            elif key.endswith("transformer.tokens_embed.weight"):
                embedding_weight_name = key
                break
            elif key.endswith("encoder.embedding.weight"):
                embedding_weight_name = key
                num_answers = int(target_file_path.split('_')[-2])
                break
            elif key.endswith("wte.weight"):
                embedding_weight_name = key
        assert embedding_weight_name != None
        weight_tensor = state_dict[embedding_weight_name]
        embedding = nn.Embedding.from_pretrained(weight_tensor).to("cpu")
    else:
        with open(id2word_path, 'rb') as f:
            id2word = pickle.load(f)
            word2id = {v: k for k, v in id2word.items()}

        with open(pretrained_wv_path, 'rb') as f:
            weight_tensor = to_var(torch.FloatTensor(pickle.load(f)))
        embedding = nn.Embedding.from_pretrained(weight_tensor, freeze=False).to("cpu")

    with codecs.open(target_file_path, "r", "utf-8") as csv_f:
        for line in csv_f:
            conv_idx = int(line.strip().split()[-1])
            if conv_idx_match != conv_idx:
                print("What?!")
                return
            conv_idx_match += 1
            context_utter = csv_f.readline().strip()

            answers = list()
            for _ in range(num_answers):
                answers.append(csv_f.readline().strip())
            
            if '<eos>' in answers[-1]:
                top_answer = answers[-1].split('<eos>')[0].strip()
            else: 
                top_answer = answers[-1].strip()

            ground_truth_utter = csv_f.readline().strip()

            if '<eos>' in ground_truth_utter:
                ground_truth_utter = ground_truth_utter.split('<eos>')[0]

            length_history.append(len(top_answer.split()))

            if context_utter == "" or top_answer == "" or ground_truth_utter == "":
                continue

            dist1_list += top_answer.split()

            try: 
                ground_truth_utter_ids = vocab.encode(ground_truth_utter)
                top_answer_utter_ids = vocab.encode(top_answer)
                embedding_list.append(embedding_compute(ground_truth_utter_ids, top_answer_utter_ids, embedding))
            except ValueError:
                embedding_list.append(0)             

            try:
                bleu_list.append(bleu_compute(ground_truth_utter, top_answer))
            except ZeroDivisionError:
                bleu_list.append(0)

            try:
                rouge_history.append(rouge_compute(ground_truth_utter, top_answer))
            except ValueError:
                rouge_history.append(np.zeros(3))

    length_mat = np.array(length_history)
    bleu_mat = np.array(bleu_list)
    rouge_mat = np.stack(rouge_history, axis=0)
    embedding_mat = np.array(embedding_list)

    avg_length = np.mean(length_mat)
    avg_bleu = np.mean(bleu_mat)
    avg_rouge = np.mean(rouge_mat, axis=0)
    avg_embedding = np.mean(embedding_mat)

    stderr_bleu = sem(bleu_mat, axis=0)
    stderr_length = sem(length_mat)
    stderr_rouge = sem(rouge_mat, axis=0)
    stderr_embedding = sem(embedding_mat, axis=0)

    dist1 = dist_compute(dist1_list)

    output_str_list = list()
    output_str_list.append(["Length", avg_length, stderr_length])
    output_str_list.append(["BLEU", avg_bleu, stderr_bleu])
    output_str_list.append(["Embedding", avg_embedding, stderr_embedding])
    output_str_list.append(["Dist1", dist1, '-' ])
    for one_name, one_avg, one_stderr in zip(rouge_names(), avg_rouge, stderr_rouge):
        output_str_list.append([one_name, one_avg, one_stderr])

    output_str = tabulate.tabulate(output_str_list, headers=["Metric", "Average", "Standard Error"])
    print(output_str)


if __name__ == "__main__":
    try:
        target_file_path = sys.argv[1]
        dataset = sys.argv[2]
        if dataset == "cornell":
            id2word_path = "/data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/datasets/cornell/id2word.pkl" #sys.argv[2]
            pretrained_wv_path = "/data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/datasets/cornell/fasttext_wv.pkl" # sys.argv[3]        
        else:
            checkpoint_path = sys.argv[3]
            model_name = checkpoint_path.split('/')[-3]
    except (KeyError, IndexError):
        print("Usage: python eval.py target_file_path")

    num_turn = 2

    main()
