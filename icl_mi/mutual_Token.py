import os, sys, multiprocessing
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict

from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer
from icl_mi.QKVO_hooker import hook_qkv_and_head_outputs

NUM_CPU_CORES = multiprocessing.cpu_count()

class mutualToken():
    def __init__(self, args=None):
        if args == None:
            args = get_default_parser()
        self.seed = args.seeds
        self.model = args.model
        
        self.prompt = "female\tmiddle\t28 -> survival\nmale\tupper\t51 -> death\nmale\tlower\t21 ->"
        
        # limit_gpus([0, 1])
        limit_gpus(range(0, 1))
        # seed module
        seed_everything(self.seed)
        # load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(self.model)
        self.inputs = self.tokenizer(prompt, return_tensors="pt")


    def get_QKOV(self):
        self.QKOV = hook_qkv_and_head_outputs(self.model, self.inputs)
    
    def get_token_dict(self):
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(self.inputs['input_ids'][0])
        self.token_dict = 

    def get_model(self):
        return self.model


