#!/usr/bin/env bash
set -euo pipefail

# ===== 可按需修改 =====
MODEL_DIR="outputs/roberta_binary_e0/best_model"
INPUT_CSV="outputs/norm/dev_top10_jieba_labeled.csv"
OUT_DIR="outputs/ablation_dev"
PYTHON_BIN="python"
# ====================

mkdir -p "${OUT_DIR}"

echo "[1/4] Baseline + default(k=10, alpha=1.0)"
PYTHONPATH=. ${PYTHON_BIN} train/infer_with_nbest.py \
  --model_dir "${MODEL_DIR}" \
  --input_csv "${INPUT_CSV}" \
  --out_json "${OUT_DIR}/ablation_k10_a1.0.json" \
  --top_k 10 \
  --alpha 1.0

echo "[2/4] k=5 alpha=1.5"
PYTHONPATH=. ${PYTHON_BIN} train/infer_with_nbest.py \
  --model_dir "${MODEL_DIR}" \
  --input_csv "${INPUT_CSV}" \
  --out_json "${OUT_DIR}/ablation_k5_a1.5.json" \
  --top_k 5 \
  --alpha 1.5

echo "[3/4] k=5 alpha=2.0"
PYTHONPATH=. ${PYTHON_BIN} train/infer_with_nbest.py \
  --model_dir "${MODEL_DIR}" \
  --input_csv "${INPUT_CSV}" \
  --out_json "${OUT_DIR}/ablation_k5_a2.0.json" \
  --top_k 5 \
  --alpha 2.0

echo "[4/4] k=5 alpha=2.0 + fallback"
PYTHONPATH=. ${PYTHON_BIN} train/infer_with_nbest.py \
  --model_dir "${MODEL_DIR}" \
  --input_csv "${INPUT_CSV}" \
  --out_json "${OUT_DIR}/ablation_k5_a2.0_fb.json" \
  --top_k 5 \
  --alpha 2.0 \
  --fallback_orig \
  --w1_threshold 0.35 \
  --margin_threshold 0.08

echo "[Summary] merge all json -> csv"
${PYTHON_BIN} - << 'PY'
import os, json, glob
import pandas as pd

out_dir = "outputs/ablation_dev"
files = sorted(glob.glob(os.path.join(out_dir, "*.json")))

rows = []
for fp in files:
    d = json.load(open(fp, "r", encoding="utf-8"))
    cfg = d.get("_config", {})
    for k, v in d.items():
        if k.startswith("_"):
            continue
        rows.append({
            "file": os.path.basename(fp),
            "setting": k,
            "accuracy": v.get("accuracy"),
            "f1": v.get("f1"),
            "precision": v.get("precision"),
            "recall": v.get("recall"),
            "fallback_ratio": v.get("fallback_ratio", None),
            "top_k": cfg.get("top_k"),
            "alpha": cfg.get("alpha"),
            "fallback_orig": cfg.get("fallback_orig")
        })

df = pd.DataFrame(rows)
df = df.sort_values(["f1","accuracy"], ascending=False)
csv_path = os.path.join(out_dir, "ablation_summary.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print(f"[Done] {csv_path}")
PY

echo "[Done] All ablations finished."