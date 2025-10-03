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
