import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
import json
import argparse
from collections import namedtuple
from datetime import datetime
import matplotlib.pyplot as plt
import math

# ==============================================================================
# 1. ARGUMENT PARSING
# ==============================================================================
parser = argparse.ArgumentParser(description="MARL-IDS Training Script")
parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes.')
parser.add_argument('--minibatch', type=int, default=1024, help='Minibatch Size.')
parser.add_argument('--training_steps', type=int, default=1000, help="Training Steps' size.")
parser.add_argument('--reward_k', type=float, default=10.0, help='Reward Function value')
# Setting a default device, can be overridden if needed.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
