import os, sys, multiprocessing
import numpy as np
import torch
from torch import nn
from joblib import Parallel, delayed
from tqdm import tqdm
# from collections import defaultdict

from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus, sum_by_head
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer
from icl_mi.QKVO_hooker import hook_qkv_and_head_outputs
from icl_mi.information import calc_information as cinfo

NUM_CORES = multiprocessing.cpu_count()

class mutualToken():
    def __init__(self, args=None):
        if args == None:
            args = get_default_parser()
        self.seed = args.seeds
        self.model = args.model
        self.nbins = args.num_of_bins
        
        self.prompt = "female middle 28 -> survival\nmale upper 51 -> death\nmale lower 21 ->"
        
        # limit_gpus([0, 1])
        limit_gpus(range(0, 1))
        # seed module
        seed_everything(self.seed)
        # load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(self.model)
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt")

        # get decoded tokens
        self.decoded_tokens = self.tokenizer.convert_ids_to_tokens(self.inputs['input_ids'][0])

        # self.vs, self.os, self.information = None, None, None
    
    def extract_vectors(self, o_array, v_array):
        """
        Iterates through all heads (i) and all words (j), 
        extracting the o vector for each word j in head i 
        and all v vectors from word 1 to j in head i.

        Parameters:
        o_array (numpy array): Array of o vectors of shape (1, num_heads, num_words, num_dims)
        v_array (numpy array): Array of v vectors of shape (1, num_heads, num_words, num_dims)

        Returns:
        results (dict): A dictionary with keys indicating the head and word, 
                        and values being a tuple of (o_vector, v_vectors)
        """
        results = {}
        
        # Extract the shape info from arrays (since first dimension is 1, we use index 0)
        num_heads = o_array.shape[1]
        num_words = o_array.shape[2]
        
        # Iterate over each head and word
        for i in range(num_heads):
            for j in range(1, num_words + 1):  # 1-indexed for word loop
                # Extract o vector for word j in head i
                o_vector = o_array[0][i][j - 1]  # Subtract 1 because of 1-based indexing
                
                # Extract v vectors for all words from word 1 to word j in head i
                v_vectors = [v_array[0][i][w] for w in range(j)]  # Collect vectors from word 1 to j
                
                # Create a key to indicate the head and word
                token = self.decoded_tokens[j-1]
                key = f"h_{i + 1}_w{j}_{token}"
                
                # Store the o_vector and v_vectors as a tuple in the dictionary
                results[key] = (o_vector, v_vectors)
        
        return results

    def get_QKOV(self):
        # get QKOV of the seq
        self.QKOV_raw = hook_qkv_and_head_outputs(self.model, self.inputs)
        # get num_head, seq_len, and num_layer (num_layer, bs, num_head, seq_len, dim)
        self.num_head = self.QKOV_raw['Q'][0].shape[1] 
        self.seq_len = self.QKOV_raw['Q'][0].shape[2]
        self.num_layer = len(self.QKOV_raw['Q'])
        # create arrays for saving the data
        # self.vs, self.os, self.information = \
        #     [[[[[None] for w in range(self.seq_len)] \
        #         for l in range(self.num_head)] \
        #             for h in range(self.num_layer)] \
        #                 for _ in range(3)]
        # self.vs, self.os, self.information = np.empty((3, self.num_layer, self.num_head, self.seq_len), dtype=object)

        # activations
        # def gaussian_exp(x):
        #     return torch.exp(-x**2)
        act = nn.Softmax()
        # act = nn.Tanh()
        # apply tanch for all the layers of QKOV_raw['V'], QKOV_raw['attn_output_each_head']
        for l in range(self.num_layer):
            self.QKOV_raw['V'][l] = act(self.QKOV_raw['V'][l])
            self.QKOV_raw['attn_output_each_head'][l] = act(self.QKOV_raw['attn_output_each_head'][l])
        
        self.ov_dict_list = []
        for l in range(self.num_layer):
            self.ov_dict_list.append(self.extract_vectors(self.QKOV_raw['attn_output_each_head'][l], self.QKOV_raw['V'][l]))

        # for w in range(self.seq_len):
        #     for h in range(self.num_head):
        #         for l in range(self.num_layer):
        #             #(num_layer, bs, num_head, seq_len, dim)
        #             self.vs[l][h][w] = tanh(self.QKOV_raw['V'][l][0][h,w,:])
        #             self.os[l][h][w] = tanh(self.QKOV_raw['attn_output_each_head'][l][0][h,w,:])
        # for l in range(self.num_layer):
        #     for h in range(self.num_head):
        #         for w in range(self.seq_len):
        #             self.vs[l][h][w] = tanh(self.QKOV_raw['V'][l][0][h,w,:])
        #             self.os[l][h][w] = tanh(self.QKOV_raw['attn_output_each_head'][l][0][h,w,:])


        return self.QKOV_raw, self.ov_dict_list
    
    def get_token_dict(self):
        # self.decoded_tokens = self.tokenizer.convert_ids_to_tokens(self.inputs['input_ids'][0])
        return self.decoded_tokens

    def calc_information(self):
        """Calculate the information of each head in each layer for each token in the sequence"""

        get_information_by_layer = [cinfo.get_info_layer(self.ov_dict_list[l], self.nbins) for l in tqdm(range(self.num_layer))]

        self.information = [sum_by_head(data, self.num_head, self.decoded_tokens) for data in get_information_by_layer]

        return self.information

        # self.information = np.array(information_array).reshape(self.seq_len, self.num_layer, self.num_head)
        

    def get_information_by_token(self):
        return self.information

    def get_model(self):
        return self.model


