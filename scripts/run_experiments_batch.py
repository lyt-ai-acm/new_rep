# -*- coding: utf-8 -*-
import argparse
import glob
import os
import subprocess
import sys
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=[], help="多个数据集路径（CSV或包含train/dev/test划分的目录）")
    p.add_argument("--datasets_file", type=str, default="", help="可选：每行一个数据集路径")
    p.add_argument("--output_dir", type=str, default="outputs/experiments_multi")
    p.add_argument("--dry_run", action="store_true")
    args, unknown = p.parse_known_args()
    return args, unknown


def read_dataset_list(args) -> List[str]:
    items: List[str] = [x for x in args.datasets if x.strip()]
    if args.datasets_file:
        with open(args.datasets_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                items.append(line)
    if not items:
        raise ValueError("No datasets provided. Use --datasets or --datasets_file.")
    return items


def pick_split_csv(dataset_dir: str, split_name: str) -> str:
    direct_csv = os.path.join(dataset_dir, f"{split_name}.csv")
    if os.path.exists(direct_csv):
        return direct_csv

    nested_patterns = [
        os.path.join(dataset_dir, split_name, "*.csv"),
        os.path.join(dataset_dir, "**", f"{split_name}.csv"),
        os.path.join(dataset_dir, "**", f"*{split_name}*.csv"),
    ]
    for pattern in nested_patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return ""


def resolve_dataset(path: str) -> Dict[str, str]:
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        return {"data_path": abs_path}
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Dataset path not found: {abs_path}")

    train_csv = pick_split_csv(abs_path, "train")
    dev_csv = pick_split_csv(abs_path, "dev")
    test_csv = pick_split_csv(abs_path, "test")
    if train_csv and dev_csv and test_csv:
        return {"train_csv": train_csv, "dev_csv": dev_csv, "test_csv": test_csv}
    raise ValueError(
        f"Could not resolve train/dev/test csv from dataset directory: {abs_path}. "
        f"Expected train.csv/dev.csv/test.csv or matching csv files under split folders."
    )


def dataset_key(path: str) -> str:
    name = os.path.basename(os.path.normpath(path))
    if name.lower().endswith(".csv"):
        name = os.path.splitext(name)[0]
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe or "dataset"


def run(cmd: List[str], dry_run: bool):
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def main():
    args, pass_through = parse_args()
    datasets = read_dataset_list(args)
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in datasets:
        resolved = resolve_dataset(dataset)
        out_dir = os.path.join(args.output_dir, dataset_key(dataset))
        cmd = [sys.executable, "scripts/run_experiments.py", "--output_dir", out_dir]
        for key, value in resolved.items():
            cmd += [f"--{key}", value]
        cmd.extend(pass_through)
        if args.dry_run and "--dry_run" not in pass_through:
            cmd.append("--dry_run")
        run(cmd, dry_run=args.dry_run)

    print("[Done]", args.output_dir)


if __name__ == "__main__":
    main()
