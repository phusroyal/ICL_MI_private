import os, sys, multiprocessing
import numpy as np
import torch, json, argparse, time, random, pickle, gzip, h5py
from torch import nn
# from joblib import Parallel, delayed
from tqdm import tqdm
# from collections import defaultdict
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus, sum_by_head
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer
# from icl_mi.QKVO_hooker import hook_qkv_and_head_outputs
from icl_mi.h_hooker import GPT2WithBlockIO, GPT2WithLayerOutputs
# from icl_mi.information import calc_information as cinfo

# NUM_CORES = multiprocessing.cpu_count()

# A function that converts NumPy arrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def tensor2numpy(tensor):
	return tensor.cpu().detach().numpy()

def list_shuffle(lst, n):
    new_lsts = []
    for _ in range(n):
        lst_cp = lst.copy()
        random.shuffle(lst_cp)
        new_lsts.append(lst_cp)
    return new_lsts

def load_data_json(file):
    """Load data from json file"""
    with open(file, "r") as f:
        data = json.load(f)
        prompt_data = data["prompts_list"]
    
    prompt_ult = []
    preds_ult = []
    outputs_ult = []

    for prompts in prompt_data:
        prompt_shuffled = list_shuffle(prompts, 50)
        prompt_list = []
        preds = []
        outputs = []
        for prompt in prompt_shuffled:
            outputs.append(set([de.split()[-1] for de in prompt]))
            p = " ".join(prompt[:-1])
            last_prompt = prompt[-1].split()
            for i in range(0, len(last_prompt)-1):
                p += " " + last_prompt[i]
            prompt_list.append(p)
            preds.append(last_prompt[-1])
        prompt_ult.append(prompt_list)
        preds_ult.append(preds)
        outputs_ult.append(outputs)
    
    def flatten(lst):
        return [item for sublist in lst for item in sublist]

    return flatten(prompt_ult), flatten(preds_ult), flatten(outputs_ult)

def arrow_finder(decoded_tokens):
    indices = [i for i, s in enumerate(decoded_tokens) if "->" in s]
    indices = np.array(indices)
    return indices

def main():
    parser = argparse.ArgumentParser()

    # initialize
    parser.add_argument("--seed", type=int, default=46, help="seed to preproduce")
    parser.add_argument("--model", type=str, default="gpt2-xl", help="model's name")
    parser.add_argument("--num_bins", type=int, default=100, help="number of bins for discretization")

    args = parser.parse_args()

    # seed everything
    seed_everything(args.seed)    

    # limit gpus
    limit_gpus(range(0, 1))

    # load data
    data_paths = [
        'data/s6-HH/selection_40.json',
        'data/s5-HM/selection_6.json',
        # 'data/s4-HL/selection_43.json',
        # 'data/s3-ML/selection_22.json',
        # 'data/s2-M/selection_40.json'
        # 'data/s1-L/selection_35.json',
    ]

    for data_path in data_paths:
        prompt_list, preds, labels = load_data_json(data_path)

        save_path = data_path.split('/se')[0]+'/ent/'+data_path.split('/')[-1][:-5]+'_fa.pkl.gz'

        # load model and tokenizer
        # model, tokenizer = load_model_and_tokenizer(args.model)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        model = GPT2WithBlockIO.from_pretrained(args.model) # schema 1,2,4
        # model = GPT2WithLayerOutputs.from_pretrained(args.model) # schema 3

        # method to calculate mutual information
        # method = 'bmi'
        # needed_layer = 4

        # Activation function
        # activation act = nn.Softmax()
        # act = nn.Sigmoid()
        # act = nn.Tanh()

        # number of layers based on model
        num_layer = model.config.n_layer
        num_head = model.config.n_head
        model_dim = model.config.n_embd

        emb_data = []
        # encode prompt
        info_lst = []
        for idx, p in tqdm(enumerate(prompt_list)):
            # print(f"Processing prompt {idx}/{len(prompt_list)}...")

            inputs = tokenizer(p, return_tensors="pt")
            decoded_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            arrow_idx = arrow_finder(decoded_tokens)

            inputs = tokenizer.encode(p, return_tensors="pt")

            # Perform a forward pass
            with torch.no_grad():  # Disable gradient tracking for memory efficiency
                outputs = model(inputs)

            # Access the outputs of each arrow
            # out_list = torch.stack([tup[0].detach().reshape(-1, model_dim)[arrow_idx] \
            #                             for tup in model.block_outputs]).numpy()
            out_list = torch.stack([tup[0].detach().reshape(-1, model_dim)[arrow_idx[-1]] \
                                        for tup in model.block_outputs]).numpy()

            emb_data.append(out_list)

        # save data
        with gzip.open(save_path, 'wb') as f:
            pickle.dump(emb_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # with h5py.File(save_path, 'w') as hf:
        #     hf.create_dataset('data', data=emb_data, compression='gzip') 
        print(f"Data saved to {save_path}")

if __name__ == "__main__":
    main()