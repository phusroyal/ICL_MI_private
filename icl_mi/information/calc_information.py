import multiprocessing
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

from typing import Optional, List

from tqdm import tqdm
from time import time

import bmi
import sys
import os
import contextlib

from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import gamma, digamma
from scipy.stats import binned_statistic

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

def _get_knn_mi(x: np.array, y: np.array, n_neighbors: int, clip_negative: Optional[bool] = False) -> float:
	"""
	Compute mutual information between two continuous variables.
	Parameters
	----------
	x, y            :   (n_samples,) Samples of two continuous random variables, 
						must have an identical shape.
	n_neighbors     :   Number of nearest neighbors to search for each point, see [1].
	clip_negative   :   Whether to clip negative values to zero.

	Returns
	-------
	mi          :   Estimated mutual information. 
					If it turned out to be negative it is replace by 0.

	NOTE
	-----
	True mutual information can't be negative. If its estimate by a numerical
	method is negative, it means (providing the method is adequate) that the
	mutual information is close to 0 and replacing it by 0 is a reasonable
	strategy.

	References
	----------
	.. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
			information". Phys. Rev. E 69, 2004.
	"""
	x = tensor2numpy(x)
	y = tensor2numpy(y)

	n_samples = x.size	

	x = x.reshape((-1, 1))      # (n_samples, 1)
	y = y.reshape((-1, 1))      # (n_samples, 1)
	xy = np.hstack((x, y))      # z = (x, y) -> (n_samples, 2)

	# Here we rely on NearestNeighbors to select the fastest algorithm.
	nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)
	# ^ distance metric used is the max-norm (https://en.wikipedia.org/wiki/Chebyshev_distance)
	# dist(z2, z1) = max(||x2-x1||, ||y2-y1||)
	# the same metric is used for x and y, i.e., simply
	# dist(x2, x1) = (x2 - x1), when x is 1-dimensional

	nn.fit(xy)                              # fitting KNN on joint z = (x, y)
	radius = nn.kneighbors()[0]             # kneighbors() returns (distances, neighbors) 
											# for all samples -> (2, n_samples, n_neighbors)
											# we only use the distances as the query radii 
	radius = np.nextafter(radius[:, -1], 0)

	# KDTree is explicitly fit to allow for the querying of number of
	# neighbors within a specified radius
	kd = KDTree(x, metric="chebyshev")
	nx = kd.query_radius(x, radius, 
							count_only=True, 
							return_distance=False) # number of points in x that are within the query radius
												# -> (n_samples)
	nx = np.array(nx) - 1.0                     # (nx-1)

	kd = KDTree(y, metric="chebyshev")
	ny = kd.query_radius(y, radius, 
							count_only=True, 
							return_distance=False) # number of points in y that are within the query radius
												# -> (n_samples)
	ny = np.array(ny) - 1.0                     # (ny -1)

	mi = (
		digamma(n_samples)
		+ digamma(n_neighbors)
		- np.mean(digamma(nx + 1))
		- np.mean(digamma(ny + 1))
	)                               # I(X; Y) = ψ(S) + ψ(k) - 1/N*sum(ψ(nx) + ψ(ny))

	if clip_negative:
		return max(0, mi)
	return mi	

def _get_binned_entropy(x: np.array, num_bins: Optional[int]=10, 
                        bin_edges: Optional[np.array]=None) -> float:
	"""
	Compute entropy by discretizing the given continuous random variables.
	Discretization here is done by binning the continuous values into 
	equally-spaced descrete clusters. 

	Parameters
	----------
	x           :   (n_samples, d) the continuous random variable
					where, d \in {1, 2}. 
	num_bins    :   number of bins into which to cluster the values.
					if, d==2: we would use num_bins**2 bins.
	bin_edges   :   pre-computed bin edges for discretization of reps;
					this is usually used when global binning is employed.

	Returns
	-------
	h           :   entropy H(X)

	"""
	if x.shape[1] == 2:	
		x_1, x_2 = x[:, 0], x[:, 1]
		if bin_edges is None:
			_, bin_edges_1, bin_edges_2 = np.histogram2d(x_1, x_2, bins=num_bins)	
		# bin_edges_1, bin_edges_2 = bin_edges
		binned_x_1, binned_x_2 = np.digitize(x_1, bin_edges_1), np.digitize(x_1, bin_edges_2)
		clusters, num_c = {}, 0
		for i in np.unique(binned_x_1):
			for j in np.unique(binned_x_2):
				if((i, j) not in clusters):
					clusters[(i, j)] = num_c
					num_c += 1
		binned_x = np.array([clusters[(binned_x_1[i], binned_x_2[i])] for i in range(len(x))])
	else:
		x = x.reshape(-1)
		if bin_edges is None:
			_, bin_edges = np.histogram(x, bins=num_bins)
		binned_x = np.digitize(x, bin_edges)

	def _get_discrete_entropy(X: np.array) -> float:
		_, counts = np.unique(X, return_counts=True)
		probs = counts / len(X)
		if np.count_nonzero(probs) <= 1:
			return 0

		ent = 0.
		for i in probs:
			ent -= i * np.log(i)

		return ent

	return _get_discrete_entropy(binned_x)

def _get_binned_mi(x: np.array, y: np.array, num_bins: int, 
                   clip_negative: Optional[bool] = True) -> float:
    n_samples = x.size

    x = x.reshape((-1, 1))      # (n_samples, 1)
    y = y.reshape((-1, 1))      # (n_samples, 1)
    xy = np.hstack((x, y))      # z = (x, y) -> (n_samples, 2)

    mi = _get_binned_entropy(x, num_bins) \
        + _get_binned_entropy(y, num_bins) \
        - _get_binned_entropy(xy, num_bins)
    
    if clip_negative:
        return max(0, mi)
    return mi

def _get_bmi(x: np.array, y: np.array) -> float:
	n_samples = x.size

	x = x.reshape((-1, 1))      # (n_samples, 1)
	y = y.reshape((-1, 1))      # (n_samples, 1)

	method = 'ksg'
	if method == 'cca':
		estimator = bmi.estimators.CCAMutualInformationEstimator()
		mi = estimator.estimate(x, y)
	elif method == 'ksg':
		estimator = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,))
		mi = estimator.estimate(x, y)
	elif method == 'mine':
		with contextlib.redirect_stdout(open(os.devnull, 'w')):
			estimator = bmi.estimators.MINEEstimator()
			mi = estimator.estimate(x, y)
	elif method == 'infoNCE':
		estimator = bmi.estimators.InfoNCEEstimator()
		mi = estimator.estimate(x, y)
	
	return mi

def get_info_word(key, ov_list, num_of_bins, method):
	o_val = ov_list[0]
	v_lst = ov_list[1]

	num_neighbors = 10

	if method == 'binned':
		# info_word = [get_info_ov(v, o_val, num_of_bins) for v in v_lst]
		info_word = [_get_binned_mi(v, o_val, num_of_bins) for v in v_lst]
	elif method == 'knn':
		info_word = [_get_knn_mi(v, o_val, num_neighbors) for v in v_lst]
	elif method == 'bmi':
		info_word = [_get_bmi(v, o_val) for v in v_lst]

	# Return the key and computed values so the dict can be updated outside
	return key, info_word

def get_info_layer(dict_by_layer, num_of_bins, method):
	info_dict = {}

	# if method == 'binned':
	# # Parallelize only at the top level, and return results to update `info_dict`
	# 	results = Parallel(n_jobs=NUM_CORES)(
	# 		delayed(get_info_word)(key, dict_by_layer[key], num_of_bins, method)
	# 		for key in dict_by_layer
	# 	)
	# elif method == 'knn':
	# 	# no parallelization for knn
	# 	results = [get_info_word(key, dict_by_layer[key], num_of_bins, method) for key in dict_by_layer]

	results = Parallel(n_jobs=NUM_CORES)(
			delayed(get_info_word)(key, dict_by_layer[key], num_of_bins, method)
			for key in dict_by_layer
		)
	
	# results = [get_info_word(key, dict_by_layer[key], num_of_bins, method) for key in dict_by_layer]

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


