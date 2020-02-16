import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import torch 
import torch.nn.functional as F
import math 


def bleu_compute(ground_truth_utter, answer_sample):
    ground_truth_utter_list = ground_truth_utter.split()
    answer_sample_list = answer_sample.split()
    return sentence_bleu([ground_truth_utter_list], answer_sample_list, smoothing_function=SmoothingFunction().method7,
                         weights=[1./3, 1./3, 1./3])

def rouge_compute(ground_truth_utter, answer_sample):
    rouge = Rouge()
    scores = rouge.get_scores(ground_truth_utter, answer_sample)
    return np.array([scores[0]["rouge-l"]["p"], scores[0]["rouge-l"]["r"], scores[0]["rouge-l"]["f"]])


def rouge_names():
    return ["ROUGE-L Precision", "ROUGE-L Recall", "ROUGE-L F1"]


def embedding_compute(ground_truth_ids, answer_sample_ids, embedding):
    ground_truth_weights = embedding(torch.LongTensor(ground_truth_ids)).mean(dim=0)
    answer_sample_weights = embedding(torch.LongTensor(answer_sample_ids)).mean(dim=0)
    cosine = F.cosine_similarity(ground_truth_weights, answer_sample_weights, dim=0, eps=1e-6).tolist()
    if math.isnan(cosine):
        raise ValueError
    return cosine

def dist_compute(all_response, dist_n=1):
    if not all_response:
        return 0.0 
    ngrams = zip(*[all_response[i:] for i in range(dist_n)])
    distinct_ngrams = set(ngrams)
    return len(distinct_ngrams) / len(all_response)


