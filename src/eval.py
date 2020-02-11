import codecs
import numpy as np
import sys
from utils import bleu_compute, rouge_compute, rouge_names, to_var, embedding_compute, dist_compute
from scipy.stats import sem
import tabulate
import torch.nn as nn 
import torch 
import pickle


def main():
    bleu_list = list()
    length_history = list()
    rouge_history = list()
    embedding_list = list()
    dist1_list = list()
    conv_idx_match = 0
    convs_top_answer = list()
    convs_ground_truth = list()

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

            # for _ in range(num_turn):

            # answers = list()
            # # for _ in range(num_answers):
            # answers.append(csv_f.readline().strip())
            # top_answer = answers[-1]
            # top_answer_splited = top_answer.split()

            # ground_truth_utter = csv_f.readline().strip()
            # # ground_truth_utter_splited = ground_truth_utter.split()

            # length_history.append(len(top_answer_splited))

            # convs_top_answer.append(top_answer)
            # convs_ground_truth.append(ground_truth_utter)

            context_utter = csv_f.readline().strip()
            top_answer = csv_f.readline().strip()
            ground_truth_utter = csv_f.readline().strip()

            length_history.append(len(top_answer.split()))

            if context_utter == "" or top_answer == "" or ground_truth_utter == "":
                continue

            dist1_list += top_answer.split()

            try: 
                embedding_list.append(embedding_compute(ground_truth_utter, top_answer, word2id, embedding))
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
        id2word_path = "/data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/datasets/cornell/id2word.pkl" #sys.argv[2]
        pretrained_wv_path = "/data/private/uilab/KAIST-AI-Conversation-Model-2019-Fall-HHI/datasets/cornell/fasttext_wv.pkl" # sys.argv[3]
    except (KeyError, IndexError):
        print("Usage: python eval.py target_file_path")

    num_turn = 2
    num_answers = 5

    main()
