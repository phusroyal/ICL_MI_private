import os
import random
from typing import Iterable

import numpy as np
import torch, json

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def limit_gpus(gpu_ids: Iterable[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

def json_reader(path):
    with open(path) as f:
        return json.load(f)

def sum_by_head(info_dict, num_head, labels, decoded_tokens):
    mi_dict = {}

    for i, t in enumerate(decoded_tokens):
        if any(f in t for f in labels) or ('->' in t):
            mi_lst = []
            # gather by head
            for h in range(1, num_head+1):
                mi_lst.append(info_dict[f"h_{h}_w{i+1}_{t}"])

            # sum by head
            mi_lst = np.sum(np.array(mi_lst), axis=0)

            mi_dict[f"w{i+1}_{t}"] = mi_lst
        else:
            mi_dict[f"w{i+1}_{t}"] = None

    return mi_dict

def none_token(info_dict, num_head, labels, decoded_tokens):
    mi_dict = {}

    for i, t in enumerate(decoded_tokens):
        if any(f in t for f in labels) or ('->' in t):
            mi_dict[f"w{i+1}_{t}"] = info_dict[f"w{i+1}_{t}"]
        else:
            mi_dict[f"w{i+1}_{t}"] = None

    return mi_dict