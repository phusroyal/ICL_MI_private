import multiprocessing
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from time import time

NUM_CORES = multiprocessing.cpu_count()

import torch, json

def tensor2numpy(tensor):
	return tensor.cpu().detach().numpy()

def calc_probs(data, bins):
	bins = bins.astype(np.float32)
	num_of_bins = bins.shape[0]

	# Discretize the data using the provided bins.
	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	# Create a view of the digitized data as a contiguous array of a specific data type.
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	unique_array_data, unique_inverse_data, unique_counts_data = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_data = unique_counts_data / float(sum(unique_counts_data))

	return p_data, unique_array_data, unique_inverse_data, digitized


def calc_entropy_for_specipic_o(current_os, pv_i):
	""" Calc entropy for specipic o """
	b2 = np.ascontiguousarray(current_os).view(
		np.dtype((np.void, current_os.dtype.itemsize * current_os.shape[1])))
	unique_array, unique_inverse_o, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_current_os = unique_counts / float(sum(unique_counts))
	p_current_os = np.asarray(p_current_os, dtype=np.float64).T
	H2V = pv_i * (-np.sum(p_current_os * np.log2(p_current_os)))
	return H2V	


def calc_cond_ent(p_vs, digitized_os, unique_inverse_vs):
	""" Condition entropy of o given v """
	H2V_array = np.array(
		[calc_entropy_for_specipic_o(digitized_os[unique_inverse_vs == i, :], p_vs[i])
		 for i in range(p_vs.shape[0])]
	)
	H2V = np.sum(H2V_array)
	return H2V


def calc_info_ov(p_vs, p_os, digitized_os, unique_inverse_vs):
	HO = -np.sum(p_os * np.log2(p_os))
	H2V = calc_cond_ent(p_vs, digitized_os, unique_inverse_vs)
	IOV = HO - H2V
	return IOV

def get_info_ov(vs, os, num_of_bins):

	# convert tensors to numpy arrays
	vs = tensor2numpy(vs)
	os = tensor2numpy(os)

	bins = np.linspace(0, 1, num_of_bins)

	p_vs, unique_array_vs, unique_inverse_vs, digitized_vs = calc_probs(vs, bins)
	p_os, unique_array_os, unique_inverse_os, digitized_os = calc_probs(os, bins)

	# Calculate the mutual information metrics local_IOV (information between output and Value matrix)
	local_IOV = calc_info_ov(p_vs, p_os, digitized_os, unique_inverse_vs)

	return local_IOV


def get_info_word(key, ov_list, num_of_bins):
	o_val = ov_list[0]
	v_lst = ov_list[1]

	info_word = [get_info_ov(v, o_val, num_of_bins) for v in v_lst]

	# Return the key and computed values so the dict can be updated outside
	return key, info_word



def get_info_layer(dict_by_layer, num_of_bins):
	info_dict = {}

	# Parallelize only at the top level, and return results to update `info_dict`
	results = Parallel(n_jobs=NUM_CORES)(
	    delayed(get_info_word)(key, dict_by_layer[key], num_of_bins)
	    for key in dict_by_layer
	)

	# Update the dictionary with results from parallel jobs
	for key, info_word in results:
		info_dict[key] = info_word

	return info_dict



# def get_info_ov_torch(vs, os, num_of_bins):
#     """
#     Calculate mutual information between batches of o and v vectors using PyTorch tensors.

#     Parameters:
#     vs: Tensor of shape (N, num_dims)
#     os: Tensor of shape (N, num_dims)
#     num_of_bins: Number of bins for discretization

#     Returns:
#     info_values: Tensor of shape (N), mutual information values for each sample
#     """
#     device = vs.device

#     # Combine vs and os into a single tensor for joint discretization
#     joint_data = torch.cat([vs, os], dim=1)  # Shape: (N, 2 * num_dims)

#     # Create bins
#     bins = torch.linspace(-1, 1, steps=num_of_bins + 1, device=device)

#     # Discretize vs, os, and joint data
#     digitized_vs = torch.bucketize(vs, bins) - 1  # Adjust indices to start from 0
#     digitized_os = torch.bucketize(os, bins) - 1
#     digitized_joint = torch.bucketize(joint_data, bins) - 1

#     # Flatten the discretized data
#     vs_flat = digitized_vs  # Shape: (N, num_dims)
#     os_flat = digitized_os  # Shape: (N, num_dims)
#     joint_flat = digitized_joint  # Shape: (N, 2 * num_dims)

#     # Compute unique rows and their counts for vs
#     unique_vs, inverse_vs = torch.unique(vs_flat, return_inverse=True, dim=0)
#     counts_vs = torch.bincount(inverse_vs)
#     p_vs = counts_vs.float() / counts_vs.sum().float()

#     # Compute unique rows and their counts for os
#     unique_os, inverse_os = torch.unique(os_flat, return_inverse=True, dim=0)
#     counts_os = torch.bincount(inverse_os)
#     p_os = counts_os.float() / counts_os.sum().float()

#     # Compute unique rows and their counts for joint data
#     unique_joint, inverse_joint = torch.unique(joint_flat, return_inverse=True, dim=0)
#     counts_joint = torch.bincount(inverse_joint)
#     p_joint = counts_joint.float() / counts_joint.sum().float()

#     # Map each sample to its probability
#     p_vs_sample = p_vs[inverse_vs]
#     p_os_sample = p_os[inverse_os]
#     p_joint_sample = p_joint[inverse_joint]

#     # Compute pointwise mutual information (PMI) for each sample
#     pmi = torch.log2(p_joint_sample / (p_vs_sample * p_os_sample) + 1e-10)

#     # Aggregate PMI values per sample
#     return pmi


# # def get_info_layer_torch(o_tensor, v_padded_tensor, lengths, num_of_bins, batch_size=1024):
# #     """
# #     Compute mutual information between o and v for the entire layer in batches.

# #     Parameters:
# #     o_tensor: Tensor of shape (num_instances, num_dims)
# #     v_padded_tensor: Tensor of shape (num_instances, max_seq_len, num_dims)
# #     lengths: Tensor of shape (num_instances), indicating valid lengths in v_padded_tensor
# #     num_of_bins: Number of bins for discretization
# #     batch_size: Number of instances to process in each batch

# #     Returns:
# #     info_values: List of mutual information values for each instance
# #     """
# #     device = o_tensor.device
# #     num_instances = o_tensor.shape[0]

# #     info_values = []

# #     for start_idx in range(0, num_instances, batch_size):
# #         end_idx = min(start_idx + batch_size, num_instances)

# #         # Slice the batch
# #         o_batch = o_tensor[start_idx:end_idx]
# #         v_batch = v_padded_tensor[start_idx:end_idx]
# #         lengths_batch = lengths[start_idx:end_idx]

# #         # Create masks based on lengths
# #         max_seq_len = v_batch.shape[1]
# #         mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths_batch.unsqueeze(1)

# #         # Flatten the v_batch and o_batch for processing
# #         v_flat = v_batch[mask]  # Shape: (total_valid_v_vectors, num_dims)
# #         o_flat = o_batch.repeat_interleave(lengths_batch, dim=0)  # Shape: (total_valid_v_vectors, num_dims)

# #         # Compute mutual information in batch
# #         pmi_values_flat = get_info_ov_torch(v_flat, o_flat, num_of_bins)  # Shape: (total_valid_v_vectors)

# #         # Aggregate mutual information per instance
# #         split_sizes = lengths_batch.tolist()
# #         pmi_per_instance = torch.split(pmi_values_flat, split_sizes)

# #         # Aggregate PMI values per instance (e.g., mean)
# #         info_values_batch = [pmi.mean().item() for pmi in pmi_per_instance]

# #         info_values.extend(info_values_batch)

# #         # Optionally, free up memory
# #         del o_batch, v_batch, lengths_batch, mask, v_flat, o_flat, pmi_values_flat
# #         torch.cuda.empty_cache()

# #     return info_values

# def get_info_layer_torch(o_tensor, v_padded_tensor, lengths, keys, num_of_bins):
#     device = o_tensor.device
#     num_instances = o_tensor.shape[0]

#     info_dict = {}

#     for idx in tqdm(range(num_instances)):
#         o_vec = o_tensor[idx].unsqueeze(0)  # Shape: (1, num_dims)
#         length = lengths[idx].item()
#         v_vecs = v_padded_tensor[idx][:length]  # Shape: (length, num_dims)
#         key = keys[idx]

#         # Expand o_vec to match v_vecs
#         o_vecs_expanded = o_vec.repeat(v_vecs.size(0), 1)  # Shape: (length, num_dims)

#         # Compute MI between o_vec and each v_vec in v_vecs
#         mi_values = get_info_ov_torch(v_vecs, o_vecs_expanded, num_of_bins)  # Returns tensor of shape (length,)

#         # Convert to list of floats
#         mi_values_list = mi_values.cpu().numpy().tolist()

#         info_dict[key] = mi_values_list

#     return info_dict


