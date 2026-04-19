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
import shlex
import subprocess


def run(cmd: str):
    print("\n[RUN]", cmd)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    parts = shlex.split(cmd)
    if parts:
        exe = os.path.basename(parts[0])
        if exe.startswith("python") and "-u" not in parts:
            parts.insert(1, "-u")
    ret = subprocess.call(parts, env=env)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    os.makedirs("outputs/norm", exist_ok=True)

    run("python scripts/01_split_weibo.py --input_csv data/Weibo_senti_100k.csv --out_dir data/splits")
    run("python scripts/02_tokenize_for_lm_jieba.py --input_csv data/splits/train.csv --output_txt data/lm/corpus_jieba.txt")

    print("\n[NOTE] KenLM请在WSL/Linux执行:")
    print("bash scripts/03_train_kenlm.sh data/lm/corpus_jieba.txt models/lm_jieba_5gram")
    print("完成后回Windows继续。")

    # 若你已经有 models/lm_jieba_5gram.klm，可继续
    run(
        "python pipeline/run_jieba.py "
        "--input_csv data/splits/dev.csv "
        "--output_csv outputs/norm/dev_top10_jieba.csv "
        "--kenlm_path models/lm_jieba_5gram.klm "
        "--word_homo resources/chinese_homophone_word.txt "
        "--char_homo resources/chinese_homophone_char.txt"
    )

    run(
        "python train/train_roberta_binary.py "
        "--data_path data/Weibo_senti_100k.csv "
        "--output_dir outputs/roberta_binary_e0 "
        "--model_name hfl/chinese-roberta-wwm-ext "
        "--epochs 3 --batch_size 16 --lr 2e-5 --max_len 128 --seed 42"
    )

    run(
        "python train/infer_with_nbest.py "
        "--model_dir outputs/roberta_binary_e0/best_model "
        "--input_csv outputs/norm/dev_top10_jieba.csv "
        "--out_json outputs/roberta_binary_e0/e123_dev_metrics.json"
    )

    print("\n[Done] 全流程执行完成")


if __name__ == "__main__":
    main()
