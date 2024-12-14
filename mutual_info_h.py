import os, sys, multiprocessing
import numpy as np
import torch, json, argparse, time
from torch import nn
from joblib import Parallel, delayed
from tqdm import tqdm
# from collections import defaultdict
import torch
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus, sum_by_head
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer
# from icl_mi.QKVO_hooker import hook_qkv_and_head_outputs
from icl_mi.h_hooker import GPT2WithBlockIO, GPT2WithLayerOutputs
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

def extract_vectors(decoded_tokens, labels, o_layer_i, i_layer_0, o_layer_n):
    """
    Efficiently extract o and v vectors for each head and word using PyTorch tensors.

    Parameters:
    decoded_tokens (list): A list of tokens decoded from the input_ids
    o_layer_i (torch.Tensor): The output tensor from the i-th layer of the model, shape (1, numwords, numdims)
    i_layer_0 (torch.Tensor): The input tensor to the 0-th layer of the model, shape (1, numwords, numdims)
    o_layer_n (torch.Tensor): The output tensor from the last layer of the model, shape (1, numwords, numdims)

    Returns:
    results (dict): A dictionary with keys indicating the word, 
                    and values being a tuple of 
                    (i-th layer output, (0-th layer input, n-th layer output))
    """
    # Get the shapes of the tensors
    num_words = o_layer_i.shape[1]
    
    # Initialize a dictionary to store the results
    results = {}

    # Loop over heads and words, but use PyTorch indexing to extract o and v vectors efficiently
    for j in range(1, num_words + 1):
        # Create the dictionary key: w<word>_<token>
        token = decoded_tokens[j - 1]
        key = f"w{j}_{token}"            
        # check if token is in labels or -> in token
        if any(f in key for f in labels) or ('->' in key): 
            # Extract o vector for the current head and word
            o_vector = o_layer_i[0, j-1]  # Shape: (num_dims,)

            # Extract all v vectors for words from 1 to j in the current head
            v_vectors = (i_layer_0[0, j-1], o_layer_n[0, j-1])  # Shapes: (j, num_dims), (j, num_dims)
            
            # Store the tuple (o_vector, v_vectors) in the dictionary
            results[key] = (o_vector, v_vectors)

    return results

def extract_vectors_s4(decoded_tokens, labels, o_layer_i, i_layer_0, i1_layer0, o_layer_n):
    """
    Efficiently extract o and v vectors for each head and word using PyTorch tensors.

    Parameters:
    decoded_tokens (list): A list of tokens decoded from the input_ids
    o_layer_i (torch.Tensor): The output tensor from the i-th layer of the model, shape (1, numwords, numdims)
    i_layer_0 (torch.Tensor): The input tensor to the 0-th layer of the model, shape (1, numwords, numdims)
    o_layer_n (torch.Tensor): The output tensor from the last layer of the model, shape (1, numwords, numdims)

    Returns:
    results (dict): A dictionary with keys indicating the word, 
                    and values being a tuple of 
                    (i-th layer output, (0-th layer input, n-th layer output))
    """
    # Get the shapes of the tensors
    num_words = o_layer_i.shape[1]
    
    # Initialize a dictionary to store the results
    results = {}

    # Loop over heads and words, but use PyTorch indexing to extract o and v vectors efficiently
    for j in range(1, num_words + 1):
        # Create the dictionary key: w<word>_<token>
        token = decoded_tokens[j - 1]
        key = f"w{j}_{token}"            
        # check if token is in labels or -> in token
        if any(f in key for f in labels) or ('->' in key): 
            # Extract o vector for the current head and word
            o_vector = o_layer_i[0, j-1]  # Shape: (num_dims,)

            if j < num_words:
                # Extract all v vectors for words from 1 to j in the current head
                v_vectors = (i_layer_0[0, j-1], i1_layer0[0, j])
            else:
                v_vectors = (i_layer_0[0, j-1], o_layer_n[0, j-1])  # Shapes: (j, num_dims), (j, num_dims)
            
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
    # model, tokenizer = load_model_and_tokenizer(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2WithBlockIO.from_pretrained(args.model) # schema 1,2,4
    # model = GPT2WithLayerOutputs.from_pretrained(args.model) # schema 3

    method = 'bmi'
    # activation act = nn.Softmax()
    # act = nn.Sigmoid()
    act = nn.Tanh()
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
        inputs = tokenizer.encode(p, return_tensors="pt")

        # Perform a forward pass
        with torch.no_grad():  # Disable gradient tracking for memory efficiency
            outputs = model(inputs)

        # apply activation
        # Access the inputs and outputs of each block

        # schema 1,2,4
        in_list = []
        out_list = []
        if method == 'binned':
            for _, (block_input, block_output) in enumerate(zip(model.block_inputs, model.block_outputs)):
                in_list.append(act(block_input[0]))
                out_list.append(act(block_output[0]))
                # in_list.append(block_input[0])
                # out_list.append(block_output[0])
        elif method == 'knn' or method == 'bmi':
            for _, (block_input, block_output) in enumerate(zip(model.block_inputs, model.block_outputs)):
                in_list.append(block_input[0])
                out_list.append(block_output[0])

        # # schema 3
        # ln1 = []
        # ln2 = []
        # mlp = []
        # for layer_idx in range(len(model.h)):
        #     ln1_output = model.ln1_outputs[layer_idx]  # Output after ln_1 in layer_idx
        #     ln2_output = model.ln2_outputs[layer_idx]  # Output after ln_2 in layer_idx
        #     mlp_output = model.mlp_outputs[layer_idx]  # Output after mlp in layer_idx

        #     ln1.append(act(ln1_output))
        #     ln2.append(act(ln2_output))
        #     mlp.append(act(mlp_output))


        # get ov_dict list
        ov_dict_list = []

        # # schema 1: mi of 0th layer and last layer
        for l in range(num_layer-1):
            ov_dict_list.append(extract_vectors(decoded_tokens,
                                                labels[idx],
                                                out_list[l], 
                                                in_list[0],
                                                out_list[-1]))

        # # schema 2: mi of input of everylayer and last layer
        # for l in range(num_layer-1):
        #     ov_dict_list.append(extract_vectors(decoded_tokens,
        #                                         labels[idx],
        #                                         out_list[l], 
        #                                         in_list[l],
        #                                         out_list[-1]))

        # # schema 4: mi of 0th layer token i and 0th layer token i+1
        # for l in range(num_layer-1):
        #     ov_dict_list.append(extract_vectors_s4(decoded_tokens,
        #                                         labels[idx],
        #                                         out_list[l], 
        #                                         in_list[0],
        #                                         in_list[0],
        #                                         out_list[-1]))

        # # schema 3: mi of ln1, ln2, mlp of every layer and last layer
        # for l in range(num_layer):
        #     ov_dict_list.append(extract_vectors(decoded_tokens,
        #                                         labels[idx],
        #                                         ln2[l], 
        #                                         ln1[l],
        #                                         mlp[l]))

        # get information for each layer
        information = [cinfo.get_info_layer(ov_dict_list[l], args.num_bins, method) for l in tqdm(range(num_layer-1))]

        # information = [sum_by_head(data, num_head, labels[idx], decoded_tokens) for data in get_information_by_layer]

        # save information as a text file for each prompt 
        with open(f"data/info_15/info_15_{idx}.txt", 'w', encoding='utf-8') as f:
            json.dump(information, f, ensure_ascii=False, indent=4, default=convert_ndarray)

        info_lst.append(information)
    
    with open(f"data/info_15/info_15_total.txt", 'w', encoding='utf-8') as f:
            json.dump(information, f, ensure_ascii=False, indent=4, default=convert_ndarray)

if __name__ == "__main__":
    main()