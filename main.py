"""
RAI2117307 Deep Learning Applications - Wine Quality Project
Steps 1-3 Execution Script (Compatible with PyCharm Direct Run)
Student: Özgün Yolcu
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# -------------------------------------------------------------
# Configuration & Reproducibility
# -------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Compute Device: {device}")
print(f"[*] Python Version: {sys.version.split()[0]}")

# -------------------------------------------------------------
# Step 1: Load Data & Inspect
# -------------------------------------------------------------
print("\n" + "="*50)
print("[Step 1] Loading Wine Quality dataset from UCI (ID: 186)...")
print("="*50)

wine_dataset = fetch_ucirepo(id=186)
X = wine_dataset.data.features.copy()
y = wine_dataset.data.targets.copy()
target_col = y.columns[0]

print(f"Features matrix shape: {X.shape} (instances, features)")
print(f"Target vector shape:   {y.shape}")
print("\nFirst 5 rows of features:")
print(X.head())
print("\nFirst 5 rows of target (Quality):")
print(y.head())

# -------------------------------------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
print("\n" + "="*50)
print("[Step 2] Exploratory Data Analysis (EDA)...")
print("="*50)

print("\nSummary statistics:")
print(X.describe().T[['mean', 'std', 'min', '50%', 'max']])

mean_val = y[target_col].mean()
median_val = y[target_col].median()
print(f"\nTarget Mean:   {mean_val:.2f}")
print(f"Target Median: {median_val:.2f}")

# Plot 1: Target distribution
print("\nPlotting Plot 1: Target distribution...")
plt.figure(figsize=(8, 4))
sns.countplot(x=y[target_col], palette="Blues_d")
plt.title("Distribution of Wine Quality Scores", fontsize=13, fontweight='bold')
plt.xlabel("Quality Score (0 to 10)", fontsize=11)
plt.ylabel("Count", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot 2: Correlation Heatmap
print("Plotting Plot 2: Feature correlation heatmap...")
full_df = pd.concat([X, y], axis=1)
corr = full_df.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Physicochemical Features Correlation with Quality", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nCorrelations with Quality score (ranked):")
print(corr[target_col].sort_values(ascending=False))

# -------------------------------------------------------------
# Step 3: Data Preprocessing & Splitting (Zero Data Leakage)
# -------------------------------------------------------------
print("\n" + "="*50)
print("[Step 3] Data Preprocessing & Train/Val/Test Split...")
print("="*50)

# Check missing values
missing = X.isnull().sum().sum() + y.isnull().sum().sum()
print(f"Missing values found: {missing}")

X_raw = X.values.astype(np.float32)
y_raw = y.values.astype(np.float32)

# 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y_raw, test_size=0.30, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

print(f"Train samples: {len(X_train)} ({len(X_train)/len(X_raw)*100:.1f}%)")
print(f"Val samples:   {len(X_val)} ({len(X_val)/len(X_raw)*100:.1f}%)")
print(f"Test samples:  {len(X_test)} ({len(X_test)/len(X_raw)*100:.1f}%)")

# Scaling - Fit ONLY on Train!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler successfully fit on Train and applied to Val/Test.")

# PyTorch DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset   = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset  = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"DataLoaders ready! Batch size: {BATCH_SIZE} (Train batches: {len(train_loader)})")
print("\n[SUCCESS] Week 7 Milestones (Steps 1, 2, 3) executed completely!")
