# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="review")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    texts = df[args.text_col].fillna("").astype(str).tolist()
    y_true = df[args.label_col].astype(int).to_numpy()

    all_logits = []
    num_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), args.batch_size),
            total=num_batches,
            desc="Evaluating",
            dynamic_ncols=True,
            leave=False,
        ):
            bt = texts[i:i+args.batch_size]
            enc = tok(bt, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.cpu().numpy()
            all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits.argmax(axis=-1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
