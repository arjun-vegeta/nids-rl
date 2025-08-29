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

# --- Upgraded DQN Agent Class with Target Network ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_size, batch_size, gamma, lr, device, soft_update, tau, network_sync_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.soft_update = soft_update
        self.tau = tau
        self.network_sync_rate = network_sync_rate
        
        self.memory = ReplayMemory(replay_buffer_size)
        
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.train_step_counter = 0

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        sampled_experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*sampled_experiences))

        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        weights_batch = torch.tensor(batch.weight, device=self.device, dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.cat([s.unsqueeze(0) for s in batch.next_state]).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        target_q_values_vec = reward_batch + (self.gamma * next_q_values)
        full_target_q_values = self.policy_net(state_batch).clone().detach()
        full_target_q_values.scatter_(1, action_batch, target_q_values_vec.unsqueeze(1))
        
        loss = weighted_mse_loss(self.policy_net(state_batch), full_target_q_values, weights_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step_counter += 1
        
        # --- Conditionally update the target network ---
        if self.soft_update:
            # Soft update (Polyak averaging)
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        else:
            # Hard update
            if self.train_step_counter % self.network_sync_rate == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"Agent's policy has been copied to target.")

        return loss.item()
        
# ==============================================================================
# 3. PLOTTING FUNCTION
# ==============================================================================
def plot_loss_curves(history, save_dir):
    print("\n--- Generating Loss Curves ---")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(12, 6))
    plt.plot(history)
    plt.title("Training Loss Curve for neural network")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    
    filename = os.path.join(save_dir, 'loss_curve_neural_network.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved loss curve for neural network to {filename}")

# ==============================================================================
# 4. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    # --- GPU Configuration & Timestamped Folder Setup ---
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA not available. Using CPU.")

    timestamp = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    
    # Directory for saving models
    model_run_folder = f"training_{timestamp}"
    model_save_path = os.path.join('saved_models', model_run_folder)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    print(f"Models for this run will be saved in: {model_save_path}")

    # Directory for saving evaluation plots
    eval_run_folder = f"evaluating_{timestamp}"
    plot_save_path = os.path.join('evaluate_models', eval_run_folder)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    print(f"Evaluation plots for this run will be saved in: {plot_save_path}")

    # --- Data Loading and Subsetting ---
    print("Loading preprocessed data...")
    X_train_tensor = torch.tensor(np.load('processed_data/X_train.npy'), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.load('processed_data/y_train.npy'), dtype=torch.long)
    print(f"Using {len(X_train_tensor)} samples for training.")
    
    with open('processed_data/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    rev_label_mapping = {v: k for k, v in label_mapping.items()}
    attack_class_indices = [v for k,v in label_mapping.items() if k != 'BENIGN']

    # --- Hyperparameters ---
    soft_update = True
    N_EPISODES = args.episodes
    TRAINING_STEPS_PER_EPISODE = args.training_steps
    REPLAY_BUFFER_SIZE = math.ceil(len(X_train_tensor) * 1.5)
    MINIBATCH_SIZE = args.minibatch
    LEARNING_RATE = 0.01
    GAMMA = 0.01
    REWARD_K = args.reward_k
    NETWORK_SYNC_RATE = 1000
    TAU = 0.005
    EPS_START = 1.0
    EPS_MIN = 0.01
    EPS_DECAY = 0.95
    
    # Save hyperparameters to a JSON file
    hyperparameters = {
        "N_EPISODES": N_EPISODES,
        "TRAINING_STEPS_PER_EPISODE": TRAINING_STEPS_PER_EPISODE,
        "SUBSET_SIZE": len(X_train_tensor),
        "REPLAY_BUFFER_SIZE": REPLAY_BUFFER_SIZE,
        "MINIBATCH_SIZE": MINIBATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "GAMMA": GAMMA,
        "REWARD_K": REWARD_K,
        "NETWORK_SYNC_RATE": NETWORK_SYNC_RATE,
        "TAU": TAU,
        "EPS_START": EPS_START,
        "EPS_MIN": EPS_MIN,
        "EPS_DECAY_PER_STEP": EPS_DECAY,
        "SOFT_UPDATE": soft_update
    }
    hyperparameters_path = os.path.join(model_save_path, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as f: json.dump(hyperparameters, f, indent=4)
    print(f"Hyperparameters saved to {hyperparameters_path}")

    print("--- Initializing Neural Network ---")
    n_features = X_train_tensor.shape[1]
    n_actions = 2

    # Initialise Loss logging
    loss_history = []

    # ==========================================================================
    # NEURAL NETWORK TRAINING (OFFLINE)
    # ==========================================================================
    print("\n--- Starting Neural Network Training (Offline) ---")

    neural_net = DQNAgent(
        state_dim=n_features,
        action_dim=n_actions,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=MINIBATCH_SIZE,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        device=device,
        soft_update=soft_update,
        tau=TAU,
        network_sync_rate=NETWORK_SYNC_RATE
    )

    epsilon = EPS_START
    for i_episode in range(N_EPISODES):
        print(f"\n--- Neural Network Training Episode {i_episode+1}/{N_EPISODES} ---")
        
        # LOGGING CHANGE: Replaced tqdm with a standard loop and periodic logging.
        print(f"Populating replay buffer with {len(X_train_tensor)} samples...")
        total_samples = len(X_train_tensor)
        # Log progress every 10%
        log_interval_samples = max(1, total_samples // 10) 
        
        for i in range(total_samples):
            state, true_label = X_train_tensor[i], y_train_tensor[i].item()
            action_tensor = neural_net.select_action(state.unsqueeze(0).to(device), epsilon)
            # reward = 1.0 if action_tensor.item() == true_label else -1.0
            
            
            if true_label == 0:
                reward = 1.0 if action_tensor.item() == true_label else -1.0
            else:
                reward = REWARD_K if action_tensor.item() == true_label else -REWARD_K
            
            
            weight = abs(reward)
            
            #true_label_map = "Malicious" if true_label == 1 else "BENIGN"
            #action_tensor_map = "Malicious" if action_tensor.item() == 1 else "BENIGN"
            #print(f"True label is {true_label_map} but predicted {action_tensor_map} and got reward {reward} with weight {weight}")
            neural_net.memory.push(state, action_tensor, reward, state, weight)
            
            # Print progress at specified intervals
            if (i + 1) % log_interval_samples == 0:
                print(f"  Buffer population: {(i + 1)} / {total_samples} samples added ({(i + 1) * 100 / total_samples:.0f}%)")

        print("\nBuffer population complete. Starting training steps for this episode.")
        
        # LOGGING CHANGE: Replaced tqdm with a standard loop and periodic logging.
        # Log progress every 10%
        log_interval_steps = max(1, TRAINING_STEPS_PER_EPISODE // 10)

        for step in range(TRAINING_STEPS_PER_EPISODE):
            loss = neural_net.train()
            if loss is not None:
                loss_history.append(loss)
            
            # Print progress at specified intervals
            if (step + 1) % log_interval_steps == 0:
                print(f"  Training step: {(step + 1)} / {TRAINING_STEPS_PER_EPISODE} completed.")
        
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        print(f"\nEpsilon Value for next episode is {epsilon} \n")

        # LOGGING CHANGE: Added end-of-episode timestamp as requested.
        end_time_str = datetime.now().strftime('%H:%M:%S_%d_%B')
        print(f"*** Episode {i_episode+1} ended at {end_time_str} ***")


    print("\n--- Neural Network Training Complete ---")

    # --- Saving Decider Model ---
    print("\n--- Saving Decider Agent Model ---")
    save_path = os.path.join(model_save_path, 'neural_net.pt')
    torch.save(neural_net.policy_net.state_dict(), save_path)
    print(f"Saved neural network model to {save_path}")
    
    print("\n--- ALL TRAINING COMPLETE ---")

    # --- Generate and Save Loss Curves ---
    plot_loss_curves(loss_history, plot_save_path)
