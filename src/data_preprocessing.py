import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from collections import Counter
from typing import List, Tuple

EXCLUDE_COLS = [
    'Reading', 'Label', 'Fault_Type', 'Severity_Level',
    'Fault_Type_Name', 'Severity_Name', 'Hierarchical_Label', 'Data_Source',
]


def load_dataset(path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    feature_columns = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLS
    ]
    for col in feature_columns:
        df[col] = df[col].fillna(df[col].median())
    return df, feature_columns


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_held = test_size + val_size
    train_df, temp_df = train_test_split(
        df, test_size=total_held, stratify=df['Fault_Type'], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['Fault_Type'], random_state=seed
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    val_df[feature_columns] = scaler.transform(val_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])
    return train_df, val_df, test_df, scaler


def build_graph(df_subset: pd.DataFrame, feature_columns: List[str], k: int = 5) -> Data:
    X = torch.tensor(df_subset[feature_columns].values, dtype=torch.float32)
    y_fault = torch.tensor(df_subset['Fault_Type'].values, dtype=torch.long)
    y_severity = torch.tensor(df_subset['Severity_Level'].values, dtype=torch.long)

    k = min(k, len(df_subset) - 1)
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn_model.fit(X.numpy())
    _, indices = nn_model.kneighbors(X.numpy())

    src, tgt = [], []
    for i in range(len(df_subset)):
        for j in range(1, k + 1):
            src.append(i)
            tgt.append(indices[i, j])

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    return Data(x=X, edge_index=edge_index, y_fault=y_fault, y_severity=y_severity)


def compute_class_weights(labels: list, n_classes: int, cap: float = 3.0) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = [
        min(total / (n_classes * counts.get(i, 1)), cap)
        for i in range(n_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)
