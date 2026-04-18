# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True, help="data/Weibo_senti_100k.csv")
    ap.add_argument("--out_dir", type=str, default="data/splits")
    ap.add_argument("--text_col", type=str, default="review")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--dev_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
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

    train.to_csv(os.path.join(args.out_dir, "train.csv"), index=False, encoding="utf-8-sig")
    dev.to_csv(os.path.join(args.out_dir, "dev.csv"), index=False, encoding="utf-8-sig")
    test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False, encoding="utf-8-sig")
    print(f"[Done] train={len(train)} dev={len(dev)} test={len(test)}")


if __name__ == "__main__":
    main()