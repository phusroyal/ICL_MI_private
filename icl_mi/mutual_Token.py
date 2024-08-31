import os, sys, multiprocessing
import numpy as np
from torch import nn
# from joblib import Parallel, delayed
# from collections import defaultdict

from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer
from icl_mi.QKVO_hooker import hook_qkv_and_head_outputs
from icl_mi.information import calc_information as cinfo

# NUM_CPU_CORES = multiprocessing.cpu_count()

class mutualToken():
    def __init__(self, args=None):
        if args == None:
            args = get_default_parser()
        self.seed = args.seeds
        self.model = args.model
        self.nbins = args.num_of_bins
        
        self.prompt = "female\tmiddle\t28 -> survival\nmale\tupper\t51 -> death\nmale\tlower\t21 ->"
        
        # limit_gpus([0, 1])
        limit_gpus(range(0, 1))
        # seed module
        seed_everything(self.seed)
        # load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(self.model)
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt")

        self.vs, self.os, self.information = None, None, None
    
    def get_QKOV(self):
        # get QKOV of the seq
        self.QKOV_raw = hook_qkv_and_head_outputs(self.model, self.inputs)
        # get num_head, seq_len, and num_layer (num_layer, bs, num_head, seq_len, dim)
        self.num_head = self.QKOV_raw['Q'][0].shape[1] 
        self.seq_len = self.QKOV_raw['Q'][0].shape[2]
        self.num_layer = len(self.QKOV_raw['Q'])
        # create arrays for saving the data
        self.vs, self.os, self.information = \
            [[[[[None] for w in range(self.seq_len)] \
                for l in range(self.num_head)] \
                    for h in range(self.num_layer)] \
                        for _ in range(3)]
        # Extract Vs and Os
        softmax = nn.Softmax()
        tanh = nn.Tanh()
        for w in range(self.seq_len):
            for h in range(self.num_head):
                for l in range(self.num_layer):
                    #(num_layer, bs, num_head, seq_len, dim)
                    self.vs[l][h][w] = tanh(self.QKOV_raw['V'][l][0][h,w,:])
                    self.os[l][h][w] = tanh(self.QKOV_raw['attn_output_each_head'][l][0][h,w,:])
    
    # def get_token_dict(self):
    #     decoded_tokens = self.tokenizer.convert_ids_to_tokens(self.inputs['input_ids'][0])
    #     self.token_dict = None

    def calc_information(self):
        """Calculate the information of each head in each layer for each token in the sequence"""
        self.information = np.array(
            [cinfo.get_info(self.vs[l][h][w], self.os[l][h][w], self.nbins)
                for w in range(self.seq_len) for h in range(self.num_head) for l in range(self.num_layer)]
        )

    def get_model(self):
        return self.model


