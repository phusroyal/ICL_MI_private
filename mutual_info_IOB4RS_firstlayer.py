import os, sys, multiprocessing
import numpy as np
import torch, json, argparse, time
from torch import nn
from joblib import Parallel, delayed
from tqdm import tqdm
# from collections import defaultdict

from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus, sum_by_head, none_token
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer
from icl_mi.in_out_b4rs_hooker import hook_io_b4rs
from icl_mi.information import calc_information as cinfo

NUM_CORES = multiprocessing.cpu_count()

def load_data_json_folder(folder):
    """Load data from json files"""
    path_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    prompt_list = []
    for path in path_list:
        with open(path, "r") as f:
            data = json.load(f)
            # convert list to str
            prompt = " ".join(data["prompts_list"][:-1])
            # split last prompt by space
            last_prompt = data["prompts_list"][-1].split(" ")
            for i in range(0, len(last_prompt)-1):
                prompt += " " + last_prompt[i]
            prompt_list.append(prompt)
    return prompt_list

def load_data_json(file):
    """Load data from json file"""
    with open(file, "r") as f:
        data = json.load(f)
        prompt_data = data["prompts_list"]
    
    prompt_list = []
    preds = []
    labels = []
    for prompt in prompt_data:
        labels.append(set([de.split()[-1] for de in prompt]))
        p = " ".join(prompt[:-1])
        last_prompt = prompt[-1].split(" ")
        for i in range(0, len(last_prompt)-1):
            p += " " + last_prompt[i]
        prompt_list.append(p)
        preds.append(last_prompt[-1])

    return prompt_list, preds, labels

# A function that converts NumPy arrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def extract_vectors(decoded_tokens, labels, o_tensor, v_tensor):
    """
    Efficiently extract o and v vectors for word using PyTorch tensors.

    Parameters:
    o_tensor (torch.Tensor): Tensor of o vectors of shape (1, num_words, num_dims)
    v_tensor (torch.Tensor): Tensor of v vectors of shape (1, num_words, num_dims)

    Returns:
    results (dict): A dictionary with keys indicating word, 
                    and values being a tuple of (o_vector, v_vectors)
    """
    # Get the shapes of the tensors
    num_words = o_tensor.shape[1]
    
    # Initialize a dictionary to store the results
    results = {}

    # Loop over words, but use PyTorch indexing to extract o and v vectors efficiently
    for j in range(1, num_words + 1):
        # Create the dictionary key: w<word>_<token>
        token = decoded_tokens[j - 1]
        key = f"w{j}_{token}"            
        # check if token is in labels or -> in token
        if any(f in key for f in labels) or ('->' in key): 
            # Extract o vector for the current head and word
            o_vector = o_tensor[0, j-1]  # Shape: (num_dims,)

            # Extract all v vectors for words from 1 to j in the current head
            v_vectors = v_tensor[0, :j]  # Shape: (j, num_dims)
            
            # Store the tuple (o_vector, v_vectors) in the dictionary
            results[key] = (o_vector, v_vectors)

    return results


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
    data_path = 'data/info_15/selection_15.json'
    prompt_list, preds, labels = load_data_json(data_path)
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    # activation
    # act = nn.Sigmoid()
    # act = nn.Tanh()
    # act = nn.Softmax()
    # number of layers based on model
    num_layer = model.config.n_layer
    num_head = model.config.n_head

    # encode prompt
    info_lst = []
    for idx, p in enumerate(prompt_list):
        # if idx < 2:
        #     continue
        print(f"Processing prompt {idx}/{len(prompt_list)}...")

        inputs = tokenizer(p, return_tensors="pt")
        decoded_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # hook qkv and head outputs
        IO_B4RS_raw = hook_io_b4rs(model, inputs)

        # apply activation
        # print("Applying activation function")
        # start = time.time()
        # for l in range(num_layer):
        #     IO_B4RS_raw['in'][l] = act(IO_B4RS_raw['in'][l])
        #     IO_B4RS_raw['out'][l] = act(IO_B4RS_raw['out'][l])
        # print(f"Activation time: {time.time()-start}")

        # get ov_dict list
        ov_dict_list = []
        # print("Extracting vectors")
        start = time.time()
        for l in range(num_layer):
            ov_dict_list.append(extract_vectors(decoded_tokens,
                                                labels[idx],
                                                IO_B4RS_raw['out'][l],
                                                IO_B4RS_raw['in'][0]))
        # print(f"Extracting time: {time.time()-start}")

        # get information for each layer
        get_information_by_layer = [cinfo.get_info_layer(ov_dict_list[l], args.num_bins, method='bmi') for l in tqdm(range(num_layer))]

        information = [none_token(data, num_head, labels[idx], decoded_tokens) for data in get_information_by_layer]

        # save information as a text file for each prompt 
        with open(f"data/info_15/info_15_{idx}.txt", 'w', encoding='utf-8') as f:
            json.dump(information, f, ensure_ascii=False, indent=4, default=convert_ndarray)

        info_lst.append(information)
    
    with open(f"data/info_15/info_15_total.txt", 'w', encoding='utf-8') as f:
            json.dump(information, f, ensure_ascii=False, indent=4, default=convert_ndarray)

if __name__ == "__main__":
    main()