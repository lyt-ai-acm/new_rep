# -*- coding: utf-8 -*-
import argparse
import os

try:
    from train._generic_training import (
        ALL_BACKBONES,
        TRANSFORMER_BACKBONES,
        load_or_split_data,
        resolve_model_name,
        save_metrics,
        set_global_seed,
        train_classical,
        train_hf,
    )
except ModuleNotFoundError:
    from _generic_training import (
        ALL_BACKBONES,
        TRANSFORMER_BACKBONES,
        load_or_split_data,
        resolve_model_name,
        save_metrics,
        set_global_seed,
        train_classical,
        train_hf,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, required=True, choices=ALL_BACKBONES)
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--train_csv", type=str, default="")
    p.add_argument("--dev_csv", type=str, default="")
    p.add_argument("--test_csv", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name", type=str, default="")
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
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)

    if not (args.data_path or (args.train_csv and args.dev_csv and args.test_csv)):
        raise ValueError("Either --data_path or all of --train_csv/--dev_csv/--test_csv must be provided")

    train_df, dev_df, test_df = load_or_split_data(
        text_col=args.text_col,
        label_col=args.label_col,
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        test_size=args.test_size,
        dev_size=args.dev_size,
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        test_csv=args.test_csv,
    )

    if args.backbone in TRANSFORMER_BACKBONES:
        model_name = resolve_model_name(args.backbone, args.model_name)
        train_hf(
            model_name=model_name,
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            text_col=args.text_col,
            label_col=args.label_col,
            output_dir=args.output_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            fp16=args.fp16,
            use_contrastive=False,
            scl_weight=0.0,
            scl_temperature=0.07,
        )
    else:
        result = train_classical(
            backbone=args.backbone,
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            text_col=args.text_col,
            label_col=args.label_col,
            output_dir=args.output_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            contrastive_weight=0.0,
            contrastive_temperature=0.07,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
        )
        save_metrics(args.output_dir, result.dev, result.test, result.confusion_matrix)

    print("[Done]", args.output_dir)


if __name__ == "__main__":
    main()
