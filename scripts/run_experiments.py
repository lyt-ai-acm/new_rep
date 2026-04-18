# -*- coding: utf-8 -*-
import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split


BACKBONES: List[Dict[str, str]] = [
    {"key": "textcnn", "name": "TextCNN", "model_name": ""},
    {"key": "bilstm_attn", "name": "BiLSTM-Attn", "model_name": ""},
    {"key": "roberta_wwm_ext", "name": "RoBERTa-wwm-ext", "model_name": "hfl/chinese-roberta-wwm-ext"},
    {"key": "macbert", "name": "MacBERT", "model_name": "hfl/chinese-macbert-base"},
    {"key": "chinesebert", "name": "ChineseBERT-base", "model_name": "shannonai/ChineseBERT-base"},
    {"key": "ernie_3_zh", "name": "ERNIE-3.0-base-zh", "model_name": "nghuyong/ernie-3.0-base-zh"},
    {"key": "erlangshen_deberta_v2", "name": "Erlangshen-DeBERTa-v2", "model_name": "IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"},
    {"key": "skep", "name": "SKEP", "model_name": "baidu/bce-ernie-1.0-skep-zh"},
]


def run(cmd: List[str], dry_run: bool = False):
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_get_f1(metrics: Dict) -> float:
    if "test" in metrics and isinstance(metrics["test"], dict):
        if "eval_f1" in metrics["test"]:
            return float(metrics["test"]["eval_f1"])
        if "f1" in metrics["test"]:
            return float(metrics["test"]["f1"])
    if "E4_Entropy_Dynamic_Gating" in metrics:
        return float(metrics["E4_Entropy_Dynamic_Gating"].get("f1", 0.0))
    return 0.0


def ensure_consistent_splits(args, split_dir: str):
    os.makedirs(split_dir, exist_ok=True)
    train_path = os.path.join(split_dir, "train.csv")
    dev_path = os.path.join(split_dir, "dev.csv")
    test_path = os.path.join(split_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path):
        return train_path, dev_path, test_path

    df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df = df[df[args.text_col] != ""].copy()
    df[args.label_col] = df[args.label_col].astype(int)

    train_dev, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df[args.label_col]
    )
    dev_ratio = args.dev_size / (1.0 - args.test_size)
    train, dev = train_test_split(
        train_dev, test_size=dev_ratio, random_state=args.seed, stratify=train_dev[args.label_col]
    )

    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    dev.to_csv(dev_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    return train_path, dev_path, test_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/Weibo_senti_100k.csv")
    p.add_argument("--nbest_csv", type=str, default="")
    p.add_argument("--output_dir", type=str, default="outputs/experiments")
    p.add_argument("--text_col", type=str, default="review")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--dev_size", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scl_weight", type=float, default=0.1)
    p.add_argument("--scl_temperature", type=float, default=0.07)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    split_dir = os.path.join(args.output_dir, "shared_splits")
    train_csv, dev_csv, test_csv = ensure_consistent_splits(args, split_dir)

    summary = {
        "config": vars(args),
        "shared_splits": {"train_csv": train_csv, "dev_csv": dev_csv, "test_csv": test_csv},
        "results": [],
    }

    for cfg in BACKBONES:
        key = cfg["key"]
        name = cfg["name"]
        print(f"\n===== {name} ({key}) =====")

        base_dir = os.path.join(args.output_dir, key, "baseline")
        ours_dir = os.path.join(args.output_dir, key, "ours")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(ours_dir, exist_ok=True)

        base_cmd = [
            sys.executable,
            "train/train_baseline.py",
            "--backbone", key,
            "--train_csv", train_csv,
            "--dev_csv", dev_csv,
            "--test_csv", test_csv,
            "--output_dir", base_dir,
            "--text_col", args.text_col,
            "--label_col", args.label_col,
            "--max_len", str(args.max_len),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--warmup_ratio", str(args.warmup_ratio),
            "--seed", str(args.seed),
        ]
        if cfg["model_name"]:
            base_cmd += ["--model_name", cfg["model_name"]]

        ours_cmd = [
            sys.executable,
            "train/train_contrastive.py",
            "--backbone", key,
            "--train_csv", train_csv,
            "--dev_csv", dev_csv,
            "--test_csv", test_csv,
            "--output_dir", ours_dir,
            "--text_col", args.text_col,
            "--label_col", args.label_col,
            "--max_len", str(args.max_len),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--warmup_ratio", str(args.warmup_ratio),
            "--seed", str(args.seed),
            "--scl_weight", str(args.scl_weight),
            "--scl_temperature", str(args.scl_temperature),
        ]
        if cfg["model_name"]:
            ours_cmd += ["--model_name", cfg["model_name"]]

        run(base_cmd, dry_run=args.dry_run)
        run(ours_cmd, dry_run=args.dry_run)

        fusion_path = os.path.join(ours_dir, "fusion_metrics.json")
        if args.nbest_csv:
            infer_cmd = [
                sys.executable,
                "train/infer_nbest_entropy.py",
                "--model_dir", os.path.join(ours_dir, "best_model"),
                "--input_csv", args.nbest_csv,
                "--out_json", fusion_path,
                "--fallback_orig",
            ]
            run(infer_cmd, dry_run=args.dry_run)

        if args.dry_run:
            bare_f1 = 0.0
            ours_f1 = 0.0
            delta = 0.0
        else:
            base_metrics = read_json(os.path.join(base_dir, "metrics.json"))
            ours_metrics = read_json(os.path.join(ours_dir, "metrics.json"))
            bare_f1 = maybe_get_f1(base_metrics)
            ours_f1 = maybe_get_f1(read_json(fusion_path)) if args.nbest_csv and os.path.exists(fusion_path) else maybe_get_f1(ours_metrics)
            delta = ours_f1 - bare_f1

        summary["results"].append(
            {
                "backbone": name,
                "key": key,
                "baseline_f1": bare_f1,
                "ours_f1": ours_f1,
                "delta_f1": delta,
                "baseline_metrics_path": os.path.join(base_dir, "metrics.json"),
                "ours_metrics_path": os.path.join(ours_dir, "metrics.json"),
                "fusion_metrics_path": fusion_path if args.nbest_csv else "",
            }
        )

    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[Done]", out_path)


if __name__ == "__main__":
    main()
