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