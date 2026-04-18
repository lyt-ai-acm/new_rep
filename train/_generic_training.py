# -*- coding: utf-8 -*-
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

TRANSFORMER_BACKBONES: Dict[str, str] = {
    "roberta_wwm_ext": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base",
    "chinesebert": "shannonai/ChineseBERT-base",
    "ernie_3_zh": "nghuyong/ernie-3.0-base-zh",
    "erlangshen_deberta_v2": "IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese",
    "skep": "baidu/bce-ernie-1.0-skep-zh",
}

ALL_BACKBONES: List[str] = ["textcnn", "bilstm_attn", *list(TRANSFORMER_BACKBONES.keys())]


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def compute_metrics_np(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def hf_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=1),
        "precision": precision_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, average="binary", pos_label=1, zero_division=0),
    }


def resolve_model_name(backbone: str, model_name: str = "") -> str:
    if model_name:
        return model_name
    if backbone not in TRANSFORMER_BACKBONES:
        raise ValueError(f"Backbone '{backbone}' has no default HF model_name")
    return TRANSFORMER_BACKBONES[backbone]


def load_or_split_data(
    text_col: str,
    label_col: str,
    data_path: str,
    output_dir: str,
    seed: int,
    test_size: float,
    dev_size: float,
    train_csv: str = "",
    dev_csv: str = "",
    test_csv: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_csv and dev_csv and test_csv:
        train = pd.read_csv(train_csv, encoding="utf-8-sig")
        dev = pd.read_csv(dev_csv, encoding="utf-8-sig")
        test = pd.read_csv(test_csv, encoding="utf-8-sig")
    else:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        df[text_col] = df[text_col].fillna("").astype(str).str.strip()
        df = df[df[text_col] != ""].copy()
        df[label_col] = df[label_col].astype(int)

        train_dev, test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df[label_col]
        )
        dev_ratio = dev_size / (1.0 - test_size)
        train, dev = train_test_split(
            train_dev, test_size=dev_ratio, random_state=seed, stratify=train_dev[label_col]
        )

    for split_name, split_df in [("train", train), ("dev", dev), ("test", test)]:
        split_df[text_col] = split_df[text_col].fillna("").astype(str).str.strip()
        split_df[label_col] = split_df[label_col].astype(int)
        split_df.to_csv(os.path.join(output_dir, f"{split_name}.csv"), index=False, encoding="utf-8-sig")

    return train, dev, test


class CharVocab:
    def __init__(self):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.stoi = {self.pad: 0, self.unk: 1}

    def build(self, texts: List[str], min_freq: int = 1):
        freq = {}
        for t in texts:
            for ch in t:
                freq[ch] = freq.get(ch, 0) + 1
        for ch, c in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            if c >= min_freq and ch not in self.stoi:
                self.stoi[ch] = len(self.stoi)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(ch, 1) for ch in text[:max_len]]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids

    @property
    def size(self) -> int:
        return len(self.stoi)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)


class CharTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, vocab: CharVocab, max_len: int):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_labels: int = 2, kernel_sizes=(3, 4, 5), num_filters=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, return_features: bool = False):
        x = self.embedding(input_ids).transpose(1, 2)
        feats = [torch.max(F.relu(conv(x)), dim=2).values for conv in self.convs]
        feat = torch.cat(feats, dim=1)
        logits = self.fc(self.dropout(feat))
        return (logits, feat) if return_features else logits


class BiLSTMAttnClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int = 2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, return_features: bool = False):
        x = self.embedding(input_ids)
        h, _ = self.lstm(x)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        feat = torch.sum(h * w.unsqueeze(-1), dim=1)
        logits = self.fc(self.dropout(feat))
        return (logits, feat) if return_features else logits


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """Compute supervised contrastive loss on batch representations.

    Steps:
    1) L2-normalize features.
    2) Build pairwise similarity logits with temperature scaling.
    3) Keep positives (same label, excluding self-pairs).
    4) Average log-probability over positives for each sample.
    """
    device = features.device
    batch_size = features.shape[0]

    features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0,
    )
    mask = mask * logits_mask

    sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - sim_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mask_sum = mask.sum(1)
    # No positive pairs in a batch position is possible; clamp denominator to keep loss finite.
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    return -mean_log_prob_pos.mean()


@dataclass
class ClassicalTrainResult:
    dev: Dict[str, float]
    test: Dict[str, float]
    confusion_matrix: List[List[int]]


def train_classical(
    backbone: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_dir: str,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.07,
    embed_dim: int = 256,
    hidden_dim: int = 256,
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = CharVocab()
    vocab.build(train_df[text_col].astype(str).tolist())

    train_ds = CharTextDataset(train_df, text_col, label_col, vocab, max_len)
    dev_ds = CharTextDataset(dev_df, text_col, label_col, vocab, max_len)
    test_ds = CharTextDataset(test_df, text_col, label_col, vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if backbone == "textcnn":
        model = TextCNNClassifier(vocab_size=vocab.size, embed_dim=embed_dim)
    elif backbone == "bilstm_attn":
        model = BiLSTMAttnClassifier(vocab_size=vocab.size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported classical backbone: {backbone}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    def eval_loader(loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                preds = torch.argmax(logits, dim=-1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
        return np.array(y_true), np.array(y_pred)

    best_dev_f1 = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, feats = model(input_ids, return_features=True)
            loss = ce_loss(logits, labels)
            if contrastive_weight > 0:
                loss = loss + contrastive_weight * supervised_contrastive_loss(feats, labels, contrastive_temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_true, dev_pred = eval_loader(dev_loader)
        dev_f1 = f1_score(dev_true, dev_pred, average="binary", pos_label=1)
        if dev_f1 > best_dev_f1:
            best_dev_f1 = float(dev_f1)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    dev_true, dev_pred = eval_loader(dev_loader)
    test_true, test_pred = eval_loader(test_loader)

    dev_metrics = compute_metrics_np(dev_true, dev_pred)
    test_metrics = compute_metrics_np(test_true, test_pred)

    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
    vocab.save(os.path.join(best_dir, "vocab.json"))
    with open(os.path.join(best_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"backbone": backbone, "max_len": max_len, "embed_dim": embed_dim, "hidden_dim": hidden_dim, "num_labels": 2},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return ClassicalTrainResult(
        dev=dev_metrics,
        test=test_metrics,
        confusion_matrix=test_metrics["confusion_matrix"],
    )


class ContrastiveCELossTrainer(Trainer):
    def __init__(self, scl_weight=0.1, scl_temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scl_weight = scl_weight
        self.scl_temperature = scl_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits

        ce = nn.CrossEntropyLoss()(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        cls_embeds = outputs.hidden_states[-1][:, 0, :]
        scl = supervised_contrastive_loss(cls_embeds, labels, temperature=self.scl_temperature)
        total = ce + self.scl_weight * scl
        return (total, outputs) if return_outputs else total


def build_hf_training_args(output_dir: str, lr: float, batch_size: int, epochs: int, weight_decay: float, warmup_ratio: float, seed: int, fp16: bool):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        report_to="none",
        seed=seed,
    )


def save_metrics(output_dir: str, dev_metrics: Dict, test_metrics: Dict, cm):
    out = {
        "dev": {k: float(v) for k, v in dev_metrics.items() if isinstance(v, (int, float))},
        "test": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        "test_confusion_matrix": cm,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def train_hf(
    model_name: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_dir: str,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    use_contrastive: bool,
    scl_weight: float,
    scl_temperature: float,
):
    from datasets import Dataset, DatasetDict

    ds = DatasetDict({
        "train": Dataset.from_pandas(
            train_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}),
            preserve_index=False,
        ),
        "dev": Dataset.from_pandas(
            dev_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}),
            preserve_index=False,
        ),
        "test": Dataset.from_pandas(
            test_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}),
            preserve_index=False,
        ),
    })

    tok = AutoTokenizer.from_pretrained(model_name)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    targs = build_hf_training_args(output_dir, lr, batch_size, epochs, weight_decay, warmup_ratio, seed, fp16)

    trainer_cls = ContrastiveCELossTrainer if use_contrastive else Trainer
    trainer_kwargs = {}
    if use_contrastive:
        trainer_kwargs.update({"scl_weight": scl_weight, "scl_temperature": scl_temperature})

    trainer = trainer_cls(
        **trainer_kwargs,
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["dev"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=hf_compute_metrics,
    )

    trainer.train()
    dev_metrics = trainer.evaluate(ds_tok["dev"])
    test_metrics = trainer.evaluate(ds_tok["test"])

    pred = trainer.predict(ds_tok["test"])
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)
    cm = confusion_matrix(y_true, y_pred).tolist()

    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)

    save_metrics(output_dir, dev_metrics, test_metrics, cm)
