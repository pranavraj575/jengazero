import torch, numpy as np, random
import sys, os


def seed(seed):
    """
    seeds pytorch and np randomness
    """
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)
