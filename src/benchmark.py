# benchmark.py
#
# This script trains and evaluates a list of classical machine learning models
# on the preprocessed IDS dataset.
#
# It creates a new timestamped directory for each run (e.g., "evaluating_models_DD-MM-YYYY_HH-MM-SS")
# and saves all results for a specific model (plots, reports) into its own
# sub-folder (e.g., "RandomForest/", "LightGBM/").

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
import time
from datetime import datetime # Added for timestamping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Import ML Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

# ==============================================================================
# PLOTTING HELPER FUNCTIONS
# We define these once to reuse them for every model
# ==============================================================================

def save_confusion_matrix(y_true, y_pred, model_name, class_names, model_output_dir):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Save with a simpler name inside the model's folder
    plt.savefig(os.path.join(model_output_dir, 'confusion_matrix.png'))
    plt.close()

def save_metrics_barchart(report_dict, model_name, model_output_dir):
    """Generates and saves a bar chart for precision, recall, and f1-score."""
    metrics_df = pd.DataFrame(report_dict).transpose()
    # Drop the summary rows for the plot
    metrics_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(18, 7), width=0.8)
    plt.title(f'{model_name} - Per-Class Performance Metrics', fontsize=20)
    plt.xlabel('Attack Class', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'class_metrics.png'))
    plt.close()

def save_fpr_barchart(cm, model_name, class_names, model_output_dir):
    """Generates and saves a bar chart for the False Positive Rate per class."""
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    # Add a small epsilon to avoid division by zero if (fp + tn) is zero
    fpr = fp / (fp + tn + 1e-9) 
    
    plt.figure(figsize=(18, 7))
    sns.barplot(x=class_names, y=fpr, palette='viridis')
    plt.title(f'{model_name} - False Positive Rate (FPR) per Class', fontsize=20)
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('FPR', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'false_positive_rate.png'))
    plt.close()

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    # --- NEW: Create timestamped parent folder ---
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    parent_output_dir = f"evaluate_models/evaluating_models_{timestamp}"
    os.makedirs(parent_output_dir, exist_ok=True)
    print(f"All evaluation results will be saved in: {parent_output_dir}")

    # --- 2. Load Data ---
    # Load the *exact same* data splits as the RL agent
    print("Loading preprocessed data...")
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    with open('processed_data/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    class_names = list(label_mapping.keys())
    rev_label_mapping = {v: k for k, v in label_mapping.items()}

    print(f"Benchmarking on {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 4. Define Models ---
    # A dictionary of models to train and evaluate
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "GaussianNB": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
        "LinearSVC": LinearSVC(max_iter=1000, dual=False, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=50, early_stopping=True, n_iter_no_change=5, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    }
