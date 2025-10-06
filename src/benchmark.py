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
