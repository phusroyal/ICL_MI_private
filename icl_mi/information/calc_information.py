import multiprocessing
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

NUM_CORES = multiprocessing.cpu_count()

# def get_information(ws, x, label, num_of_bins, interval_information_display, model, layerSize,
#                     calc_parallel=True, py_hats=0):
# 	print('Start calculating the information...')
# 	# Creates num_of_bins equally spaced bins between -1 and 1 \
# 	#  for discretizing the activations of the network layers.
# 	bins = np.linspace(-1, 1, num_of_bins)

# 	if calc_parallel:
# 		params = np.array(Parallel(n_jobs=NUM_CORES, prefer="threads"
# 		                           )(delayed(calc_information_for_epoch)
# 		                             (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
# 		                              label,
# 		                              b, b1, len(unique_a), pys,
# 		                              pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize)
# 		                             for i in range(len(ws))))
# 	else:
# 		params = np.array([calc_information_for_epoch
# 		                   (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
# 		                    label, b, b1, len(unique_a), pys,
# 		                    pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize)
# 		                   for i in range(len(ws))])
# 	return params

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
	p_data = unique_counts / float(sum(unique_counts))

	return p_data, unique_array_data, unique_inverse_data, unique_counts_data, b2

###
def calc_entropy_for_specipic_t(current_ts, px_i):
	"""Calc entropy for specipic t"""
	b2 = np.ascontiguousarray(current_ts).view(
		np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_current_ts = unique_counts / float(sum(unique_counts))
	p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
	H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
	return H2X

def calc_condtion_entropy(px, t_data, unique_inverse_x):
	# Condition entropy of t given x
	H2X_array = np.array(
		Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
		                           for i in range(px.shape[0])))
	H2X = np.sum(H2X_array)
	return H2X
###


def calc_entropy_for_specipic_o(current_os, pv_i):
	""" Calc entropy for specipic t """
	

def calc_cond_ent(p_vs, o_data, unique_array_vs):
	""" Condition entropy of o given v """
	H2V_array = np.array(
		Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
		                           for i in range(px.shape[0])))

def calc_info_ov(p_vs, p_os, bins, unique_inverse_v, unique_inverse_o):
	HO = -np.sum(p_os * np.log2(p_os))
	H2V = 

def get_info(vs, os, num_of_bins, calc_parallel=True):
	
	bins = np.linspace(-1, 1, num_of_bins)

	p_vs, unique_array_vs, unique_inverse_vs, unique_counts_vs, bv = calc_probs(vs, bins)
	p_os, unique_array_os, unique_inverse_os, unique_counts_os, bo = calc_probs(os, bins)




	if calc_parallel:
		params = np.array(Parallel(n_jobs=NUM_CORES, prefer="threads"
		                           )(delayed(calc_information_for_epoch)
		                             (i, interval_information_display, ws[i], bins, unique_inverse_x, unique_inverse_y,
		                              label,
		                              b, b1, len(unique_a), pys,
		                              pxs, p_y_given_x, pys1, model.save_file, x.shape[1], layerSize)
		                             for i in range(len(ws))))
	else:
		pass 

	
