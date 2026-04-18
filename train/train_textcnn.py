# -*- coding: utf-8 -*-
import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common_senti_train import compute_metrics, load_and_split_data, save_metrics, set_seed


class CharDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx][: self.max_len]
        token_ids = [self.vocab.get(ch, 1) for ch in text]
        pad_len = self.max_len - len(token_ids)
        if pad_len > 0:
            token_ids += [0] * pad_len
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, embed_dim: int, num_filters: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in (3, 4, 5)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 3, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)
        pooled = [torch.max(torch.relu(conv(x)), dim=-1).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def build_vocab(texts: List[str], min_freq: int) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(list(str(t)))
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for ch, freq in counter.items():
        if freq >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_labels: int):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return compute_metrics(y_true, y_pred, num_labels=num_labels)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="review")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--dev_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--num_filters", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--min_freq", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    train_df, dev_df, test_df, _, id2label = load_and_split_data(
        data_path=args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
        dev_size=args.dev_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    num_labels = len(id2label)

    vocab = build_vocab(train_df[args.text_col].astype(str).tolist(), args.min_freq)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    train_ds = CharDataset(train_df[args.text_col].astype(str).tolist(), train_df["label_id"].astype(int).tolist(), vocab, args.max_len)
    dev_ds = CharDataset(dev_df[args.text_col].astype(str).tolist(), dev_df["label_id"].astype(int).tolist(), vocab, args.max_len)
    test_ds = CharDataset(test_df[args.text_col].astype(str).tolist(), test_df["label_id"].astype(int).tolist(), vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(
        vocab_size=len(vocab),
        num_labels=num_labels,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids), labels)
            loss.backward()
            optimizer.step()
        dev_metrics = evaluate(model, dev_loader, device, num_labels)
        if dev_metrics["f1"] > best_f1:
            best_f1 = float(dev_metrics["f1"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

    dev_metrics = evaluate(model, dev_loader, device, num_labels)
    test_metrics = evaluate(model, test_loader, device, num_labels)
    save_metrics(args.output_dir, dev_metrics, test_metrics, id2label, vars(args))
    print("[Done]", args.output_dir)


if __name__ == "__main__":
    main()
