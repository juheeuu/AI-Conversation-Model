import numpy as np
import torch
import torch.nn as nn
from layers import masked_cross_entropy
from utils import to_var
import os
from tqdm import tqdm
from math import isnan
import codecs
import sys
from .solver import Solver

class SloverPretrainingBased(Solver):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SloverPretrainingBased, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)
    
    def train(self):
        pass

    def evaluate(self):
        pass 

    def test(self):
        pass 

    def export_samples(self, beam_size=5):
        pass

