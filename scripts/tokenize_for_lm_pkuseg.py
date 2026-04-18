# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_txt", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="review")
    args = ap.parse_args()

    import pkuseg
    seg = pkuseg.pkuseg()

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    texts = df[args.text_col].fillna("").astype(str).tolist()

    n = 0
    with open(args.output_txt, "w", encoding="utf-8") as f:
        for t in texts:
            t = t.strip()
            if not t:
                continue
            toks = seg.cut(t)
            if not toks:
                continue
            f.write(" ".join(toks) + "\n")
            n += 1
    print(f"[Done] lines={n} -> {args.output_txt}")


if __name__ == "__main__":
    main()