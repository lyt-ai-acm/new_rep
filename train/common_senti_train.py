# -*- coding: utf-8 -*-
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_stratify_labels(y: Sequence[int]) -> Sequence[int] | None:
    labels = pd.Series(y)
    if labels.value_counts().min() < 2:
        return None
    return y


def load_and_split_data(
    data_path: str,
    text_col: str,
    label_col: str,
    test_size: float,
    dev_size: float,
    seed: int,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, int]]:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns `{text_col}` and `{label_col}`")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].copy()
    df[label_col] = df[label_col].astype(int)

    original_labels = sorted(df[label_col].unique().tolist())
    label2id = {int(v): i for i, v in enumerate(original_labels)}
    id2label = {i: int(v) for v, i in label2id.items()}
    df["label_id"] = df[label_col].map(label2id).astype(int)

    stratify = _safe_stratify_labels(df["label_id"].tolist())
    train_dev, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    dev_ratio = dev_size / (1.0 - test_size)
    stratify_td = _safe_stratify_labels(train_dev["label_id"].tolist())
    train, dev = train_test_split(
        train_dev,
        test_size=dev_ratio,
        random_state=seed,
        stratify=stratify_td,
    )

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8-sig")
    dev.to_csv(os.path.join(output_dir, "dev.csv"), index=False, encoding="utf-8-sig")
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8-sig")
    return train, dev, test, label2id, id2label


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_labels: int) -> Dict[str, object]:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    labels = list(range(num_labels))

    if num_labels == 2:
        f1 = f1_score(y_true_np, y_pred_np, average="binary", pos_label=1, zero_division=0)
        precision = precision_score(y_true_np, y_pred_np, average="binary", pos_label=1, zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, average="binary", pos_label=1, zero_division=0)
    else:
        f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        precision = precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)

    return {
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np, labels=labels).tolist(),
    }


def save_metrics(
    output_dir: str,
    dev_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
    id2label: Dict[int, int],
    config: Dict[str, object],
) -> None:
    out = {
        "dev": dev_metrics,
        "test": test_metrics,
        "label_mapping": {str(i): v for i, v in id2label.items()},
        "config": config,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
