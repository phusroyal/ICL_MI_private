import multiprocessing
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

NUM_CORES = multiprocessing.cpu_count()

def tensor2numpy(tensor):
	return tensor.cpu().detach().numpy()

def calc_probs(data, bins):
	bins = bins.astype(np.float32)
	num_of_bins = bins.shape[0]

	# Discretize the data using the provided bins.
		# Reshape and Flatten: The data is reshaped and flattened.
		# Digitize: The np.digitize function assigns each data point to a bin.
		# Remap to Bin Values: The indices returned by np.digitize are mapped back to the corresponding bin values.
		# Reshape: The digitized data is reshaped back to match the original data structure.
	digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
	# Create a view of the digitized data as a contiguous array of a specific data type.
	b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
	#  Identify unique elements in the digitized data and compute their probabilities.
		# Unique Elements: The np.unique function identifies unique elements in b2.
		# Inverse Mapping: unique_inverse_data maps original data to its unique equivalent.
		# Counts: unique_counts stores the number of occurrences of each unique element.
		# Probabilities: p_ts is calculated as the normalized counts of the unique elements.
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
	# H2V_array = np.array(
	# 	Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_o)(digitized_os[unique_inverse_vs == i, :], p_vs[i])
	# 	                           for i in range(p_vs.shape[0])))
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

	bins = np.linspace(-1, 1, num_of_bins)

	p_vs, unique_array_vs, unique_inverse_vs, digitized_vs = calc_probs(vs, bins)
	p_os, unique_array_os, unique_inverse_os, digitized_os = calc_probs(os, bins)

	# Calculate the mutual information metrics local_IOV (information between output and Value matrix)
	local_IOV = calc_info_ov(p_vs, p_os, digitized_os, unique_inverse_vs)
	return local_IOV

def get_info_word(key, ov_list, num_of_bins):
	o_val = ov_list[0]
	v_lst = ov_list[1]

	# info_word = Parallel(n_jobs=NUM_CORES)(
	# 	delayed(get_info_ov)(v, o_val, num_of_bins)
	# 	for v in v_lst
	# )

	info_word = [get_info_ov(v, o_val, num_of_bins) for v in v_lst]

	# Return the key and computed values so the dict can be updated outside
	return key, info_word

	# return info_word


def get_info_layer(dict_by_layer, num_of_bins):
	# create a new dict and its keys are from the dict_by_layer and each key has an empty list
	# info_dict = dict.fromkeys(dict_by_layer)

	# Parallel(n_jobs=NUM_CORES)(
	# 	delayed(get_info_word)(info_dict, key, dict_by_layer[key], num_of_bins)
	# 	for key in dict_by_layer
	# )

	# for key in dict_by_layer:
	# 	get_info_word(info_dict, key, dict_by_layer[key], num_of_bins)

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
	

