import numpy as np
import random, json, os
from scipy.optimize import brentq
from sklearn.metrics import mutual_info_score
from itertools import combinations

from tqdm import tqdm

def synthesizer(f1_values, f2_values, out_values, mi_f1_output, mi_f2_output, num_demo):
    num_f1 = len(f1_values)
    num_f2 = len(f2_values)
    num_out = len(out_values)

    # Ensure variables have at least two categories
    assert num_f1 >= 2 and num_f2 >= 2 and num_out >= 2, "Each variable must have at least two categories."

    # Function to compute mutual information given theta
    def compute_mi_theta(N, theta):
        if theta == 0.0:
            return 0.0
        if theta == 1.0:
            return np.log2(N)
        P_diag = theta / N + (1 - theta) / N**2
        P_offdiag = (1 - theta) / N**2
        with np.errstate(divide='ignore', invalid='ignore'):
            I_diag = N * P_diag * np.log2(N**2 * P_diag)
            I_offdiag = N * (N - 1) * P_offdiag * np.log2(N**2 * P_offdiag)
        # Replace NaNs resulting from 0 * log(0) with zeros
        I_diag = np.nan_to_num(I_diag)
        I_offdiag = np.nan_to_num(I_offdiag)
        I = I_diag + I_offdiag
        return I

    # Function to find theta for desired MI
    def find_theta_for_mi(N, mi_target):
        max_mi = np.log2(N)
        if mi_target > max_mi:
            raise ValueError(f"mi_target {mi_target} exceeds maximum possible mutual information {max_mi:.4f} for N={N}")
        if mi_target == 0:
            return 0.0
        if mi_target == max_mi:
            return 1.0

        def func(theta):
            mi = compute_mi_theta(N, theta)
            return mi - mi_target

        # Slightly adjust the interval to avoid endpoints
        theta = brentq(func, 1e-10, 1 - 1e-10)
        return theta

    # Use the smaller number between features and outputs for matching indices
    N_f1_out = min(num_f1, num_out)
    N_f2_out = min(num_f2, num_out)

    # Find theta values for desired mutual information
    theta_f1 = find_theta_for_mi(N_f1_out, mi_f1_output)
    theta_f2 = find_theta_for_mi(N_f2_out, mi_f2_output)

    # Build joint probability tables
    joint_probs_f1_out = np.full((num_f1, num_out), (1 - theta_f1) / (num_f1 * num_out))
    for i in range(N_f1_out):
        joint_probs_f1_out[i, i] += theta_f1 / N_f1_out

    joint_probs_f2_out = np.full((num_f2, num_out), (1 - theta_f2) / (num_f2 * num_out))
    for i in range(N_f2_out):
        joint_probs_f2_out[i, i] += theta_f2 / N_f2_out

    # Normalize the joint probabilities
    joint_probs_f1_out /= joint_probs_f1_out.sum()
    joint_probs_f2_out /= joint_probs_f2_out.sum()

    # Generate counts from joint probabilities
    counts_f1_out = (joint_probs_f1_out * num_demo).astype(int)
    counts_f2_out = (joint_probs_f2_out * num_demo).astype(int)

    # Adjust counts to sum to num_demo
    counts_f1_out.flat[-1] += num_demo - counts_f1_out.sum()
    counts_f2_out.flat[-1] += num_demo - counts_f2_out.sum()

    # Generate data for feature1 and output
    f1_list = []
    out_list = []
    for i in range(num_f1):
        for j in range(num_out):
            count = counts_f1_out[i, j]
            f1_list.extend([f1_values[i]] * count)
            out_list.extend([out_values[j]] * count)

    # Shuffle the data
    combined = list(zip(f1_list, out_list))
    random.shuffle(combined)
    f1_list, out_list = zip(*combined)

    # Generate data for feature2 and output
    f2_list = []
    out_list_f2 = []
    for i in range(num_f2):
        for j in range(num_out):
            count = counts_f2_out[i, j]
            f2_list.extend([f2_values[i]] * count)
            out_list_f2.extend([out_values[j]] * count)

    # Shuffle the data
    combined_f2 = list(zip(f2_list, out_list_f2))
    random.shuffle(combined_f2)
    f2_list, out_list_f2 = zip(*combined_f2)

    # Align feature2 with the original output_list
    # Create a mapping from output to feature2
    output_to_f2 = {out: [] for out in out_values}
    for out, f2 in zip(out_list_f2, f2_list):
        output_to_f2[out].append(f2)

    # Build the aligned feature2 list
    f2_list_aligned = []
    output_counts = {out: 0 for out in out_values}
    for out in out_list:
        if output_counts[out] < len(output_to_f2[out]):
            idx = output_counts[out]
            f2_list_aligned.append(output_to_f2[out][idx])
            output_counts[out] += 1
        else:
            # If we run out, assign a random feature2 value
            f2_list_aligned.append(random.choice(f2_values))

    # Compute actual mutual information
    mi_f1_actual = mutual_info_score(f1_list, out_list) / np.log(2)  # Convert to bits
    mi_f2_actual = mutual_info_score(f2_list_aligned, out_list) / np.log(2)  # Convert to bits

    return list(f1_list), list(f2_list_aligned), list(out_list), [mi_f1_actual, mi_f2_actual]

# Check and format MI_list
def validate_mi_list(mi_list):
    """Ensure that MI_list contains pairs of mutual information values."""
    if not all(isinstance(item, tuple) and len(item) == 2 for item in mi_list):
        raise ValueError("MI_list must contain tuples of two elements each.")
    return mi_list

def select_keys_no_overlap(feature_dict, num_selections=10):
    """Function to select 2 unique keys for features and 1 key for output with no overlapping selections."""
    # Get all possible combinations of 2 keys for features and 1 for output
    available_keys = list(feature_dict.keys())
    
    # Generate all possible unique triplets (2 feature keys, 1 output key)
    possible_combinations = list(combinations(available_keys, 3))
    
    # Shuffle to randomize the selections
    random.shuffle(possible_combinations)
    
    # Select the first `num_selections` combinations
    selected_combinations = possible_combinations[:num_selections]

    # Now, break the triplets into feature1, feature2, and output keys
    selected_keys = [(comb[0], comb[1], comb[2]) for comb in selected_combinations]

    return selected_keys

from collections import Counter
import random

def remove_excess_duplicates_and_shuffle(input_list, max_occurrences=2):
    # Count the occurrences of each element in the list
    element_count = Counter(input_list)
    
    result_list = []
    
    # Loop through the list and add elements to result list if they have appeared less than or equal to the allowed occurrences
    for item in input_list:
        if element_count[item] > max_occurrences:
            if result_list.count(item) < max_occurrences:
                result_list.append(item)
        else:
            result_list.append(item)
    
    # Shuffle the result list to ensure duplicates are not next to each other
    random.shuffle(result_list)
    
    return result_list

def simplifier(prompts_list, selected_keys, feature_dict, index):
    """Function to simplify the prompts for each selection."""
    # reduce duplicates when only the number of dubplicates is greater than 3
    prompts_list = remove_excess_duplicates_and_shuffle(prompts_list, max_occurrences=2)

    # get feature1, feature2, and output keys
    f1s = []
    f2s = []
    outputs = []
    for i in range(len(prompts_list)):
        f1s.append(prompts_list[i].split()[0])
        f2s.append(prompts_list[i].split()[1])
        outputs.append(prompts_list[i].split()[3])

    mi_f1_actual = mutual_info_score(f1s, outputs) / np.log(2)  # Convert to bits
    mi_f2_actual = mutual_info_score(f2s, outputs) / np.log(2)  # Convert to bits

    return_prompts = []
    return_prompts.append(prompts_list)
    f1_dict_list = feature_dict[selected_keys[index][0]]
    f2_dict_list = feature_dict[selected_keys[index][1]]
    output_dict_list = feature_dict[selected_keys[index][2]]
    for l in range(1, 11):
        f1s_new = []
        f2s_new = []
        outputs_new = []
        for f1, f2, output in zip(f1s, f2s, outputs):
            # get index of the feature1, feature2, and output keys
            f1_index = f1_dict_list.index(f1)
            f2_index = f2_dict_list.index(f2)
            output_index = output_dict_list.index(output)
            
            f1s_new.append(feature_dict[selected_keys[index-l][0]][f1_index])
            f2s_new.append(feature_dict[selected_keys[index-l][1]][f2_index])
            outputs_new.append(feature_dict[selected_keys[index-l][2]][output_index])

        return_prompts.append([f"{f1} {f2} -> {out}\n" for f1, f2, out in zip(f1s_new, f2s_new, outputs_new)])
    return return_prompts, [mi_f1_actual, mi_f2_actual]



def generate_meta_data(mi_list, feature_dict, num_demo=300, seed=42):
    """Generate meta data dictionary with selected features, mutual information, and synthesized data."""
    # Validate MI_list format
    mi_list = validate_mi_list(mi_list)

    # Select unique feature and output keys without overlap
    selected_keys = select_keys_no_overlap(feature_dict, num_selections=len(mi_list))
    
    meta_data = {}

    # For each mutual information pair and feature key selection
    for i, (mi_f1, mi_f2) in tqdm(enumerate(mi_list)):
        # Select feature1, feature2, and output keys
        f1_key, f2_key, output_key = selected_keys[i]

        # Extract corresponding values from the FEATURE_DICT
        feature1 = feature_dict[f1_key]
        feature2 = feature_dict[f2_key]
        output = feature_dict[output_key]

        # Run the synthesizer function to generate the lists
        f1_list, f2_list, out_list, actual_mi = synthesizer(feature1, feature2, output, mi_f1, mi_f2, num_demo)

        prompts = [f"{f1} {f2} -> {out}\n" for f1, f2, out in zip(f1_list, f2_list, out_list)]

        new_prompts, actual_mi = simplifier(prompts, selected_keys, feature_dict, i)

        # Store the metadata
        meta_data[f"selection_{i+1}"] = {
            "features": {
                "feature1": f1_key,
                "feature2": f2_key,
                "output": output_key
            },
            "mi": actual_mi,
            "synthesized_data": {
                "feature1_list": f1_list,
                "feature2_list": f2_list,
                "output_list": out_list
            }, 
            "prompts_list": new_prompts
        }
    
    return meta_data

def save_meta_data_to_files(meta_data, directory="data_files"):
    """Save the meta_data dictionary to individual JSON files."""
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save each selection's metadata to a separate JSON file
    for key, value in meta_data.items():
        file_path = os.path.join(directory, f"{key}.json")
        
        # Write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(value, json_file, indent=4)

    print(f"Meta data saved to {directory}/")

def generate_mi_pairs(num_pairs=10, min_mi=0.1, max_mi=3.0):
    """Generate a list of MI pairs where the first MI is at least 1.5 the second MI."""
    mi_pairs = []
    
    for _ in range(num_pairs):
        # Randomly generate the second MI within the range [min_mi, max_mi/3]
        mi2 = random.uniform(min_mi, max_mi / 3)
        
        # Ensure the first MI is at least 1.5 the second MI
        mi1 = random.uniform(mi2, max_mi/2)
        
        mi_pairs.append((mi2, mi1))
    
    return mi_pairs


if __name__ == "__main__":
    # FEATURE_DICT = {
    #     "fruits": ["apple", "lime", "melon", "orange", "corn", "fig", "berry", "date"],
    #     "colors": ["red", "green", "blue", "yellow", "black", "grey", "white", "brown"],
    #     "animals": ["dog", "cat", "bird", "fish", "ant", "ape", "bat", "frog"],
    #     "elements": ["fire", "water", "earth", "air", "light", "ice", "metal", "wood"],
    #     "vehicles": ["car", "bike", "bus", "train", "plane", "boat", "ship", "truck"],
    #     "cutlery": ["fork", "knife", "pot", "pick", "plate", "bowl", "cup", "pan"],
    #     "clothes": ["shirt", "pants", "hat", "watch", "dress", "ring", "boot", "suits"],
    #     "metals": ["iron", "copper", "zinc", "gold", "silver", "nickel", "tin", "iodine"]
    # }
    FEATURE_DICT = {
        "fruits": ["apple", "lime", "melon", "date"],
        "colors": ["red", "green", "blue", "yellow"],
        "animals": ["dog", "cat", "bird", "fish"],
        "elements": ["fire", "water", "earth", "air"],
        "vehicles": ["car", "bike", "bus", "train"],
        "cutlery": ["fork", "knife", "pot", "pick"],
        "clothes": ["shirt", "pants", "hat", "watch"],
        "metals": ["iron", "copper", "zinc", "gold"]
    }

    seed = 42
    num_pairs = 20
    num_demo = 100 #to fit context length of 1042

    random.seed(seed)
    np.random.seed(seed)

    MI_list = generate_mi_pairs(num_pairs=num_pairs, min_mi=0.1, max_mi=2.0)
    print(MI_list)
    meta_data = generate_meta_data(MI_list, FEATURE_DICT, num_demo=num_demo, seed=seed)
    save_meta_data_to_files(meta_data, directory="data/data_files")