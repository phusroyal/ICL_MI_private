import os, sys, multiprocessing
import numpy as np
import torch, json, argparse, time
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
    labels = []
    for prompt in prompt_data:
        p = " ".join(prompt[:-1])
        last_prompt = prompt[-1].split(" ")
        for i in range(0, len(last_prompt)-1):
            p += " " + last_prompt[i]
        prompt_list.append(p)
        labels.append(last_prompt[-1])

    return prompt_list, labels

# A function that converts NumPy arrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def extract_vectors(decoded_tokens, o_tensor, v_tensor):
    """
    Efficiently extract o and v vectors for each head and word using PyTorch tensors.

    Parameters:
    o_tensor (torch.Tensor): Tensor of o vectors of shape (1, num_heads, num_words, num_dims)
    v_tensor (torch.Tensor): Tensor of v vectors of shape (1, num_heads, num_words, num_dims)

    Returns:
    results (dict): A dictionary with keys indicating the head and word, 
                    and values being a tuple of (o_vector, v_vectors)
    """
    # Get the shapes of the tensors
    num_heads = o_tensor.shape[1]
    num_words = o_tensor.shape[2]
    
    # Initialize a dictionary to store the results
    results = {}

    # Loop over heads and words, but use PyTorch indexing to extract o and v vectors efficiently
    for i in range(num_heads):
        for j in range(1, num_words + 1):
            # Extract o vector for the current head and word
            o_vector = o_tensor[0, i, j - 1]  # Shape: (num_dims,)

            # Extract all v vectors for words from 1 to j in the current head
            v_vectors = v_tensor[0, i, :j]  # Shape: (j, num_dims)

            # Create the dictionary key: h_<head>_w<word>_<token>
            token = decoded_tokens[j - 1]
            key = f"h_{i + 1}_w{j}_{token}"
            
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
    data_path = 'data/selection_15.json'
    prompt_list, labels = load_data_json(data_path)
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    # activation
    act = nn.Softmax()
    # number of layers based on model
    num_layer = model.config.n_layer
    num_head = model.config.n_head

    # encode prompt
    info_lst = []
    for idx, p in enumerate(prompt_list):
        print(f"Processing prompt {idx+1}/{len(prompt_list)}...")

        inputs = tokenizer(p, return_tensors="pt")
        decoded_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # hook qkv and head outputs
        QKOV_raw = hook_qkv_and_head_outputs(model, inputs)

        # apply activation
        # print("Applying activation function")
        start = time.time()
        for l in range(num_layer):
            QKOV_raw['V'][l] = act(QKOV_raw['V'][l])
            QKOV_raw['attn_output_each_head'][l] = act(QKOV_raw['attn_output_each_head'][l])
        # print(f"Activation time: {time.time()-start}")

        # get ov_dict list
        ov_dict_list = []
        # print("Extracting vectors")
        start = time.time()
        for l in range(num_layer):
            ov_dict_list.append(extract_vectors(decoded_tokens,
                                                QKOV_raw['attn_output_each_head'][l], 
                                                QKOV_raw['V'][l]))
        # print(f"Extracting time: {time.time()-start}")

        # get information for each layer
        get_information_by_layer = [cinfo.get_info_layer(ov_dict_list[l], args.num_bins) for l in tqdm(range(num_layer))]

        information = [sum_by_head(data, num_head, decoded_tokens) for data in get_information_by_layer]

        # save information as a text file for each prompt
        with open(f"data/info_15_{idx}.txt", 'w', encoding='utf-8') as f:
            json.dump(information, f, ensure_ascii=False, indent=4, default=convert_ndarray)

        info_lst.append(information)
    
    with open(f"data/info_15_total.txt", 'w', encoding='utf-8') as f:
            json.dump(information, f, ensure_ascii=False, indent=4, default=convert_ndarray)

if __name__ == "__main__":
    main()