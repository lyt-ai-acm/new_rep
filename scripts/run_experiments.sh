#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-data/Weibo_senti_100k.csv}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LEN=${MAX_LEN:-128}
LR=${LR:-2e-5}
SEED=${SEED:-42}

declare -A MODELS=(
  [roberta]="hfl/chinese-roberta-wwm-ext"
  [ernie]="nghuyong/ernie-3.0-base-zh"
  [deberta]="IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"
)

for key in "${!MODELS[@]}"; do
  model_name="${MODELS[$key]}"

  echo "[Baseline][$key] CE only"
  python train/train_contrastive.py \
    --data_path "${DATA_PATH}" \
    --output_dir "outputs/${key}_baseline" \
    --model_name "${model_name}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_len "${MAX_LEN}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --scl_weight 0.0

  echo "[Ours][$key] CE + SupCon"
  python train/train_contrastive.py \
    --data_path "${DATA_PATH}" \
    --output_dir "outputs/${key}_ours" \
    --model_name "${model_name}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_len "${MAX_LEN}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --scl_weight 0.1 \
    --scl_temperature 0.07

  echo "[Info] metrics:"
  echo "  outputs/${key}_baseline/metrics.json"
  echo "  outputs/${key}_ours/metrics.json"
done
