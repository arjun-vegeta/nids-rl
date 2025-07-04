# Network Intrusion Detection System (NIDS) Implementation

This repository contains the code and implementation for a Multi-Agent Reinforcement Learning (MARL) based Network Intrusion Detection System (NIDS-RL). The project encompasses data preprocessing, MARL agent training, and performance benchmarking against various classical and ensemble ML baselines on NIDS datasets (e.g., NITK-NIDS / CIC-IDS).

## Repository Structure

```text
NIDS_Implementation/
├── src/                     # Source code for training and evaluation
│   ├── benchmark.py         # Trains and evaluates a suite of classical ML models
│   ├── evaluate.py          # Script for evaluating trained models
│   └── train.py             # Script for model training
├── notebooks/               # Jupyter Notebooks for exploration and data prep
│   └── preprocessing_data.ipynb
├── data/                    # [Ignored by Git] Raw datasets
├── processed_data/          # [Ignored by Git] Processed numpy arrays (.npy)
├── saved_models/            # [Ignored by Git] Trained model binaries
├── evaluate_models/         # [Ignored by Git] Evaluation results and plots
└── archives/                # [Ignored by Git] Old backups and zips
```

## Getting Started

### 1. Data and Models
The raw datasets and pre-trained models are not provided publicly and must be supplied locally. **Note:** The dataset used for this project belongs to NITK (National Institute of Technology Karnataka) and cannot be made public.

**Instructions:**
1. Place your raw dataset `.csv` files in the `data/` directory.
2. If you have any pre-trained models, place them in the `saved_models/` directory.

### 2. Preprocessing
To preprocess the raw data, execute the Jupyter notebook located in `notebooks/`. It will read the `.csv` files from the `data/` directory and output balanced, processed `.npy` arrays (e.g. `X_train.npy`, `y_train.npy`) into the `processed_data/` directory.

### 3. Model Training & Benchmarking
All scripts are designed to be run from the **root directory** of the project.

To run a full benchmark of multiple classical ML models (Random Forest, LightGBM, Logistic Regression, etc.):
```bash
python src/benchmark.py
```
This script will automatically load data from `processed_data/` and save evaluation metrics, classification reports, and confusion matrices into timestamped folders inside `evaluate_models/`.

To train the MARL agent, run:
```bash
python src/train.py --episodes 100 --minibatch 1024
```
This will create a timestamped folder inside `saved_models/` containing the model binaries.

To evaluate a trained MARL model, you must provide the exact folder name where the model is saved:
```bash
python src/evaluate.py --folder_name <timestamped_model_folder_name>
```

## Requirements
The following dependencies are required to run the scripts and notebook:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `torch`
- `tqdm`


## Team
This repository represents the Major Project developed by:
- **Arjun R**
- **Hari Hardhik Reddy**
- **Samrudh M**
