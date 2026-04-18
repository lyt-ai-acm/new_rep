# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd

from pipeline.segmenters import JiebaSegmenter
from pipeline.normalize_top10 import HomophoneNormalizer, NormalizeConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="review")
    ap.add_argument("--kenlm_path", type=str, required=True)
    ap.add_argument("--word_homo", type=str, required=True)
    ap.add_argument("--char_homo", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    cfg = NormalizeConfig()
    norm = HomophoneNormalizer(
        kenlm_path=args.kenlm_path,
        word_homo_path=args.word_homo,
        char_homo_path=args.char_homo,
        segmenter=JiebaSegmenter(),
        cfg=cfg,
    )

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    texts = df[args.text_col].fillna("").astype(str).tolist()

    rows = []
    for i, t in enumerate(texts):
        topn = norm.generate_topn(t)
        row = {"id": i, "orig": t}
        for k, item in enumerate(topn, start=1):
            row[f"cand_{k}"] = item["text"]
            row[f"w_{k}"] = item["weight"]
            row[f"score_{k}"] = item["final_score"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[Done] -> {args.output_csv}")


if __name__ == "__main__":
    main()