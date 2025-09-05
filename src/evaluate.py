# evaluate.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
import pandas as pd
import random
from collections import namedtuple, deque
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. AGENT AND MODEL DEFINITIONS (Required for loading weights)
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

# --- Dummy Replay Memory and DQNAgent (Architecture needed for initialization) ---
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "weight"])

class ReplayMemory: # Dummy version for initialization
    def __init__(self, capacity): pass
    def push(self, *args): pass
    def sample(self, batch_size): pass
    def __len__(self): return 0

class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)

# ==============================================================================
# 2. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARL-IDS Evaluation Script")
    parser.add_argument('--folder_name', type=str, required=True, help='Folder name to evaluate models from.')
    args = parser.parse_args()

    device = 'cuda:0'
    
    # --- 1. Setup and Initialization ---
    print("--- 1. Initializing Setup ---")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    #Folder name we are pulling models from
    run_folder_name = "_" + args.folder_name

    #Create input directory for plots
    input_dir = os.path.join('saved_models/', 'training' + run_folder_name)

    # Create output directory for plots
    
    output_dir = os.path.join('evaluate_models/', 'evaluating' + run_folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load test data and label mapping
    X_test_tensor = torch.tensor(np.load('processed_data/X_test.npy'), dtype=torch.float32)
    y_test = np.load('processed_data/y_test.npy')
    with open('processed_data/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    rev_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = list(label_mapping.keys())
    n_features = X_test_tensor.shape[1]
    n_actions = 2
    n_classes = len(class_names)

    # subset_size = 50000
    # X_test_tensor = X_test_tensor[:subset_size]
    # y_test = y_test[:subset_size]
    # print(f"Using a subset of the data for evaluating: {len(X_test_tensor)} samples.")

    # --- 2. Initialize Architectures and Load Trained Models ---
    print(f"\n--- 2. Loading Trained Models from this directory {input_dir} ---")

    # Load Decider Agent
    neural_net_loaded = DQNAgent(state_dim=n_features, action_dim=n_actions, device=device)
    load_path = os.path.join(input_dir, 'neural_net.pt')
    neural_net_loaded.policy_net.load_state_dict(torch.load(load_path, map_location=device))
    neural_net_loaded.policy_net.eval()
    print("Loaded neural network.")
