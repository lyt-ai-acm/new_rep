import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)

    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--orig_col", type=str, default="orig")
    p.add_argument("--cand_prefix", type=str, default="cand_")
    p.add_argument("--w_prefix", type=str, default="w_")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--top_k", type=int, default=10, help="使用前K个候选")
    p.add_argument("--alpha", type=float, default=1.0, help="权重温度幂次: w^alpha 后再归一化")
    return p.parse_args()


def predict_prob(texts, tokenizer, model, device, batch_size=64, max_len=128):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def normalize_weights(W, alpha=1.0):
    W = np.clip(W, 1e-12, None)
    if alpha != 1.0:
        W = np.power(W, alpha)
    Z = W.sum(axis=1, keepdims=True)
    Z = np.where(Z <= 0, 1.0, Z)
    return W / Z


def compute_normalized_entropy(W):
    W_safe = np.clip(W, 1e-12, 1.0)
    H = -np.sum(W_safe * np.log(W_safe), axis=1)
    return H / np.log(W.shape[1] + 1e-12)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    if args.label_col not in df.columns:
        raise KeyError(f"label col '{args.label_col}' not found in {args.input_csv}")

    cand_cols_all = [c for c in df.columns if c.startswith(args.cand_prefix)]
    w_cols_all = [c for c in df.columns if c.startswith(args.w_prefix)]
    if len(cand_cols_all) == 0 or len(w_cols_all) == 0:
        raise ValueError("No candidate or weight columns found.")

    max_cand = min(len(cand_cols_all), len(w_cols_all))
    K = min(args.top_k, max_cand)

    cand_cols = [f"{args.cand_prefix}{i}" for i in range(1, K + 1)]
    w_cols = [f"{args.w_prefix}{i}" for i in range(1, K + 1)]

    y_true = df[args.label_col].astype(int).to_numpy()

    if args.orig_col in df.columns:
        orig_texts = df[args.orig_col].astype(str).tolist()
    elif "review" in df.columns:
        orig_texts = df["review"].astype(str).tolist()
    else:
        raise ValueError("Missing original text column. Need --orig_col or review column.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)

    p_orig = predict_prob(orig_texts, tokenizer, model, device, args.batch_size, args.max_len)

    P = []
    for c in cand_cols:
        texts = df[c].fillna("").astype(str).tolist()
        p = predict_prob(texts, tokenizer, model, device, args.batch_size, args.max_len)
        P.append(p)
    P = np.stack(P, axis=1)

    W = df[w_cols].fillna(0.0).astype(float).to_numpy()
    Wn = normalize_weights(W, alpha=args.alpha)

    p_e1 = P[:, 0]
    y_e1 = (p_e1 >= 0.5).astype(int)

    p_e2 = P.mean(axis=1)
    y_e2 = (p_e2 >= 0.5).astype(int)

    p_fusion = (P * Wn).sum(axis=1)
    y_e3 = (p_fusion >= 0.5).astype(int)

    lambda_entropy = np.clip(compute_normalized_entropy(Wn), 0.0, 1.0)
    p_e4 = lambda_entropy * p_orig + (1.0 - lambda_entropy) * p_fusion
    y_e4 = (p_e4 >= 0.5).astype(int)

    result = {
        "Base_orig": compute_metrics(y_true, (p_orig >= 0.5).astype(int)),
        "E1_top1": compute_metrics(y_true, y_e1),
        "E2_topk_avg": compute_metrics(y_true, y_e2),
        "E3_topk_weighted": compute_metrics(y_true, y_e3),
        "E4_entropy_soft_blend": compute_metrics(y_true, y_e4),
        "_config": {
            "top_k": K,
            "alpha": args.alpha,
            "mean_normalized_entropy": float(lambda_entropy.mean()),
            "mean_lambda": float(lambda_entropy.mean())
        }
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[Done] {args.out_json}")


if __name__ == "__main__":
    main()
