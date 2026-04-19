# -*- coding: utf-8 -*-
"""
三分类评测脚本（独立使用）
- 输出 Accuracy / Macro-F1 / 每类P-R-F1 / 混淆矩阵
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    texts = df[args.text_col].fillna("").astype(str).tolist()
    y_true = df[args.label_col].astype(int).to_numpy()

    logits_all = []
    num_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), args.batch_size),
            total=num_batches,
            desc="Evaluating",
            dynamic_ncols=True,
            leave=False,
        ):
            bt = texts[i:i + args.batch_size]
            enc = tok(bt, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            logits_all.append(logits.cpu().numpy())

    logits_all = np.concatenate(logits_all, axis=0)
    y_pred = np.argmax(logits_all, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    out = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "neg": {"p": float(p[0]), "r": float(r[0]), "f1": float(f1[0]), "support": int(sup[0])},
        "neu": {"p": float(p[1]), "r": float(r[1]), "f1": float(f1[1]), "support": int(sup[1])},
        "pos": {"p": float(p[2]), "r": float(r[2]), "f1": float(f1[2]), "support": int(sup[2])},
        "confusion_matrix": cm,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
