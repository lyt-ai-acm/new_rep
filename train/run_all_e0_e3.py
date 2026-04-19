# -*- coding: utf-8 -*-
"""
一站式串行执行（Python版编排）:
1) split
2) jieba分词语料
3) (提示) 训练KenLM
4) 生成Top10
5) 训练E0
6) 评测E1/E2/E3
"""
import os
import subprocess
import argparse


def run(cmd: str):
    print("\n[RUN]", cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--dev_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--split_dir", type=str, default="data/splits")
    parser.add_argument("--norm_output_csv", type=str, default="outputs/norm/dev_top10_jieba.csv")
    parser.add_argument("--lm_corpus_txt", type=str, default="data/lm/corpus_jieba.txt")
    parser.add_argument("--kenlm_path", type=str, default="models/lm_jieba_5gram.klm")
    parser.add_argument("--output_dir", type=str, default="outputs/roberta_binary_e0")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.norm_output_csv), exist_ok=True)

    if args.train_csv and args.dev_csv and args.test_csv:
        train_csv, dev_csv = args.train_csv, args.dev_csv
    else:
        if not args.data_path:
            raise ValueError("Either --data_path or all of --train_csv/--dev_csv/--test_csv must be provided")
        run(f"python scripts/split_weibo.py --input_csv {args.data_path} --out_dir {args.split_dir}")
        train_csv = os.path.join(args.split_dir, "train.csv")
        dev_csv = os.path.join(args.split_dir, "dev.csv")

    run(f"python scripts/tokenize_for_lm_jieba.py --input_csv {train_csv} --output_txt {args.lm_corpus_txt}")

    print("\n[NOTE] KenLM请在WSL/Linux执行:")
    print(f"bash scripts/train_kenlm_Version2.sh {args.lm_corpus_txt} {os.path.splitext(args.kenlm_path)[0]}")
    print("完成后回Windows继续。")

    # 若你已经有 models/lm_jieba_5gram.klm，可继续
    run(
        "python pipeline/run_jieba.py "
        f"--input_csv {dev_csv} "
        f"--output_csv {args.norm_output_csv} "
        f"--kenlm_path {args.kenlm_path} "
        "--word_homo resources/chinese_homophone_word.txt "
        "--char_homo resources/chinese_homophone_char.txt"
    )

    run(
        "python train/train_roberta_binary.py "
        + (f"--data_path {args.data_path} " if args.data_path else "")
        + (f"--train_csv {args.train_csv} --dev_csv {args.dev_csv} --test_csv {args.test_csv} " if args.train_csv and args.dev_csv and args.test_csv else "")
        + f"--output_dir {args.output_dir} "
        + f"--model_name {args.model_name} "
        "--epochs 3 --batch_size 16 --lr 2e-5 --max_len 128 --seed 42"
    )

    run(
        "python train/infer_with_nbest.py "
        f"--model_dir {args.output_dir}/best_model "
        f"--input_csv {args.norm_output_csv} "
        f"--out_json {args.output_dir}/e123_dev_metrics.json"
    )

    print("\n[Done] 全流程执行完成")


if __name__ == "__main__":
    main()
