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
