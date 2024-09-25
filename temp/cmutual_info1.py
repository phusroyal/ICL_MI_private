import sys
import multiprocessing
import numpy as np
import torch
import json
import argparse
import time
import os
from torch import nn
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Assuming necessary imports from your custom modules
from icl_mi.utils.exp_params import get_default_parser
from icl_mi.utils.misc import seed_everything, limit_gpus, sum_by_head
from icl_mi.utils.model_tok_loader import load_model_and_tokenizer
from icl_mi.QKVO_hooker import hook_qkv_and_head_outputs
from icl_mi.information import calc_information as cinfo

NUM_CORES = multiprocessing.cpu_count()

def load_data_json(folder):
    """Load data from json files"""
    path_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    prompt_list = []
    for path in path_list:
        with open(path, "r") as f:
            data = json.load(f)
            # Construct prompt
            prompt = " ".join(data["prompts_list"][:-1])
            last_prompt = data["prompts_list"][-1].split(" ")
            prompt += " " + " ".join(last_prompt[:-1])
            prompt_list.append(prompt)
    return prompt_list

# def extract_vectors(o_tensor, v_tensor):
#     """
#     Prepare tensors for batch processing.

#     Parameters:
#     o_tensor (torch.Tensor): Tensor of shape (1, num_heads, num_words, num_dims)
#     v_tensor (torch.Tensor): Tensor of shape (1, num_heads, num_words, num_dims)

#     Returns:
#     o_tensor: Tensor of shape (num_heads * num_words, num_dims)
#     v_padded_tensor: Tensor of shape (num_heads * num_words, max_seq_len, num_dims)
#     lengths: Tensor of shape (num_heads * num_words), indicating valid lengths in v_padded_tensor
#     """
#     # Remove batch dimension
#     o_tensor = o_tensor[0]  # Shape: (num_heads, num_words, num_dims)
#     v_tensor = v_tensor[0]  # Shape: (num_heads, num_words, num_dims)

#     num_heads, num_words, num_dims = v_tensor.shape

#     # Prepare o_tensor
#     o_tensor = o_tensor.view(-1, num_dims)  # Shape: (num_heads * num_words, num_dims)

#     # Prepare v_padded_tensor
#     max_seq_len = num_words
#     v_padded_tensor = torch.zeros(num_heads, num_words, max_seq_len, num_dims, device=v_tensor.device)

#     # Create v_padded_tensor where for each position j, it contains v_vectors from positions 0 to j
#     for j in range(num_words):
#         v_padded_tensor[:, j, :j+1, :] = v_tensor[:, :j+1, :]

#     # Reshape v_padded_tensor to (num_heads * num_words, max_seq_len, num_dims)
#     v_padded_tensor = v_padded_tensor.view(-1, max_seq_len, num_dims)

#     # Create lengths tensor indicating valid lengths in v_padded_tensor
#     lengths = torch.arange(1, num_words+1, device=v_tensor.device).unsqueeze(0).expand(num_heads, -1).contiguous().view(-1)

#     return o_tensor, v_padded_tensor, lengths

def extract_vectors(decoded_tokens, o_tensor, v_tensor):
    # Remove batch dimension
    o_tensor = o_tensor[0]  # Shape: (num_heads, num_words, num_dims)
    v_tensor = v_tensor[0]  # Shape: (num_heads, num_words, num_dims)

    num_heads, num_words, num_dims = v_tensor.shape

    # Prepare o_tensor
    o_tensor = o_tensor.view(-1, num_dims)  # Shape: (num_heads * num_words, num_dims)

    # Prepare v_padded_tensor
    max_seq_len = num_words
    v_padded_tensor = torch.zeros(num_heads, num_words, max_seq_len, num_dims, device=v_tensor.device)

    # Create v_padded_tensor where for each position j, it contains v_vectors from positions 0 to j
    for j in range(num_words):
        v_padded_tensor[:, j, :j+1, :] = v_tensor[:, :j+1, :]

    # Reshape v_padded_tensor to (num_heads * num_words, max_seq_len, num_dims)
    v_padded_tensor = v_padded_tensor.view(-1, max_seq_len, num_dims)

    # Create lengths tensor indicating valid lengths in v_padded_tensor
    lengths = torch.arange(1, num_words+1, device=v_tensor.device).unsqueeze(0).expand(num_heads, -1).contiguous().view(-1)

    # Create keys
    keys = []
    for i in range(num_heads):
        for j in range(num_words):
            token = decoded_tokens[j]
            key = f"w{j+1}_{token}"
            keys.append(key)

    return o_tensor, v_padded_tensor, lengths, keys



def main():
    parser = argparse.ArgumentParser()

    # Initialize arguments
    parser.add_argument("--seed", type=int, default=46, help="seed to reproduce")
    parser.add_argument("--model", type=str, default="gpt2-xl", help="model's name")
    parser.add_argument("--num_bins", type=int, default=100, help="number of bins for discretization")
    parser.add_argument("--data_folder", type=str, default="data/data_files", help="folder containing data")

    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed)
    # Limit GPUs
    limit_gpus(range(0, 1))
    # Load data
    prompt_list = load_data_json(args.data_folder)
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    # Activation function
    act = nn.Softmax(dim=-1)
    # Number of layers and heads based on model
    num_layer = model.config.n_layer
    num_head = model.config.n_head

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encode prompts
    info_lst_setence = []
    for p in tqdm(prompt_list):
        inputs = tokenizer(p, return_tensors="pt").to(device)
        decoded_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Hook qkv and head outputs
        QKOV_raw = hook_qkv_and_head_outputs(model, inputs)

        # Apply activation function
        print("Applying activation function")
        start = time.time()
        for l in range(num_layer):
            QKOV_raw['V'][l] = act(QKOV_raw['V'][l])
            QKOV_raw['attn_output_each_head'][l] = act(QKOV_raw['attn_output_each_head'][l])
        print(f"Activation time: {time.time() - start:.2f}s")

        # # Get ov_dict list
        # ov_dict_list = []
        # print("Extracting vectors")
        # start = time.time()
        # o_tensor, v_padded_tensor, lengths = extract_vectors(
        #                                                         QKOV_raw['attn_output_each_head'][l], 
        #                                                         QKOV_raw['V'][l]
        #                                                     )
        # print(f"Extracting time: {time.time() - start:.2f}s")

        # # Compute information for each layer
        # print("Computing information")
        # start = time.time()
        # get_information_by_layer = get_info_layer_torch(
        #                                                 o_tensor, v_padded_tensor, lengths, args.num_bins
        #                                             )
        # information = [sum_by_head(data, num_head, decoded_tokens) for data in get_information_by_layer]
        # print(f"Information computation time: {time.time() - start:.2f}s")

        # info_lst.append(information)

        # In your main loop, replace the per-layer processing with:
        info_lst = []
        # for l in range(num_layer):
        #     print(f"Processing layer {l+1}/{num_layer}")
        #     # Extract tensors
        #     o_tensor, v_padded_tensor, lengths = extract_vectors(
        #         QKOV_raw['attn_output_each_head'][l],
        #         QKOV_raw['V'][l]
        #     )

        #     # Compute information
        #     info_values = cinfo.get_info_layer_torch(
        #         o_tensor, v_padded_tensor, lengths, args.num_bins
        #     )

        #     # Store information for the layer
        #     info_lst.append(info_values)
        for l in tqdm(range(num_layer)):
            print(f"Processing layer {l+1}/{num_layer}")

            # Extract tensors and keys
            o_tensor, v_padded_tensor, lengths, keys = extract_vectors(
                decoded_tokens,
                QKOV_raw['attn_output_each_head'][l],
                QKOV_raw['V'][l]
            )

            # Compute information
            info_dict = cinfo.get_info_layer_torch(
                o_tensor, v_padded_tensor, lengths, keys, args.num_bins
            )

            # Store information for the layer
            info_lst.append(info_dict)
            
            # save info_dict for each layer to json file
            with open(f"data/info_dict_layer_{l}.json", "w") as f:
                json.dump(info_dict, f)


    # Save information
    np.save("data/info.npy", info_lst)

if __name__ == "__main__":
    main()