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

# ==============================================================================
# 2. MODEL AND AGENT DEFINITIONS
# ==============================================================================

# --- Q-Network Definition ---
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- High-Performance NumPy Replay Buffer ---
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "weight"])

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self.current_size = 0
        self.memory = np.empty(self.capacity, dtype=object)

    def push(self, *args):
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        return self.memory[indices]

    def __len__(self):
        return self.current_size

# --- Corrected Custom Weighted MSE Loss Function ---
def weighted_mse_loss(q_values, target_q_values, weights):
    error = q_values - target_q_values
    weighted_error = error * weights
    loss = weighted_error.pow(2)
    return loss.mean()
