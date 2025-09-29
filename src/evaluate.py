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

    # --- 3. Generate Predictions on the Test Set ---
    print("\n--- 3. Generating Predictions ---")
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(len(X_test_tensor)), desc="Evaluating"):
            state = X_test_tensor[i].unsqueeze(0).to(device)
            q_values = neural_net_loaded.policy_net(state)
            prediction = q_values.argmax(1).item()
            predictions.append(prediction)
    
    y_pred = np.array(predictions)

    # --- 4. Calculate and Display Metrics in Console ---
    print("\n--- 4. Performance Metrics ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")

    labels = list(label_mapping.values())
    report_str = classification_report(y_test, y_pred, labels=labels, target_names=class_names, digits=3, zero_division=0)
    print("Classification Report:")
    print(report_str)

    # --- Save Combined Metrics to a Single IMAGE File ---

    # 1. Combine all the text you want to display into a single string.
    full_report_text = (
        f"Overall Accuracy\n"
        f"================\n"
        f"{accuracy:.4f}\n\n"
        f"Classification Report\n"
        f"======================\n"
        f"{report_str}"
    )

    # 2. Create a figure to render the text on.
    # We can adjust figsize to make sure the text fits. (width, height) in inches.
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3. Hide the axes and grid, as we only want the text.
    ax.axis('off')

    # 4. Render the text.
    # `ha='left'` and `va='top'` align the text to the top-left.
    # `family='monospace'` is CRUCIAL to keep the report columns aligned.
    ax.text(0.01, 0.99, full_report_text, transform=ax.transAxes,
            fontdict={'family': 'monospace', 'size': 10},
            ha='left', va='top')

    # 5. Save the figure as an image.
    # `bbox_inches='tight'` crops the saved image to the content.
    image_file_path = os.path.join(output_dir, 'performance_metrics.png')
    plt.savefig(image_file_path, bbox_inches='tight', dpi=300)
    plt.close() # Close the figure to free up memory

    print(f"-> Performance metrics report saved as an image to: {image_file_path}\n")
    
    # This is for saving performance metrics as .txt file
    """report_file_path = os.path.join(output_dir, 'performance_metrics.txt')
    with open(report_file_path, 'w') as f:
        # Write the Overall Accuracy section
        f.write("Overall Accuracy\n")
        f.write("================\n")
        f.write(f"{accuracy:.4f}\n\n")  # Add extra newlines for spacing
        
        # Write the Classification Report section
        f.write("Classification Report\n")
        f.write("======================\n")
        f.write(report_str)"""

    # --- 5. Generate and Save Plots ---
    print(f"\n--- 5. Generating and Saving Plots to '{output_dir}' Directory ---")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print("Saved confusion_matrix.png")

    # Plot 2: Per-Class Performance Bar Chart
    report_dict = classification_report(y_test, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    metrics_df = pd.DataFrame(report_dict).transpose()
    metrics_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(18, 7), width=0.8)
    plt.title('Per-Class Performance Metrics', fontsize=20)
    plt.xlabel('Attack Class', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics_comparison.png'))
    plt.close()
    print("Saved class_metrics_comparison.png")

    # Plot 3: False Positive Rate (FPR) Bar Chart
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    fpr = fp / (fp + tn)
    
    plt.figure(figsize=(18, 7))
    sns.barplot(x=class_names, y=fpr, palette='viridis')
    plt.title('False Positive Rate (FPR) per Class', fontsize=20)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('FPR', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'false_positive_rate.png'))
    plt.close()
    print("Saved false_positive_rate.png")

    # Plot 4: Test Set Class Distribution Chart
    class_counts = pd.Series(y_test).map(rev_label_mapping).value_counts()
    plt.figure(figsize=(18, 7))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='plasma')
    plt.title('Test Set Class Distribution', fontsize=20)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('Number of Samples', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_set_distribution.png'))
    plt.close()
    print("Saved test_set_distribution.png")

    print("\n--- Evaluation Complete ---")
