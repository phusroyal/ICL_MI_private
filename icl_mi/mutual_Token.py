import os, sys, multiprocessing
import numpy as np
from joblib import Parallel, delayed

from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer

NUM_CPU_CORES = multiprocessing.cpu_count()

class mutualToken():
    def __init__(self, args=None):
        if args == None:
            args = get_default_parser()
        self.seed = args.seeds
        self.model = args.model

        # limit_gpus([0, 1])
        limit_gpus(range(0, 1))
        # seed module
        seed_everything(self.seed)
        # load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(self.model)

    def get_model(self):
        return self.model


