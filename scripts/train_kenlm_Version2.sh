#!/usr/bin/env bash
set -e
# 用法: bash scripts/03_train_kenlm.sh data/lm/corpus_jieba.txt models/lm_jieba_5gram
CORPUS_TXT=$1
OUT_PREFIX=$2

if [ -z "$CORPUS_TXT" ] || [ -z "$OUT_PREFIX" ]; then
  echo "Usage: bash scripts/03_train_kenlm.sh <corpus_txt> <out_prefix>"
  exit 1
fi

mkdir -p "$(dirname "$OUT_PREFIX")"
lmplz -o 5 --discount_fallback < "$CORPUS_TXT" > "${OUT_PREFIX}.arpa"
build_binary -s "${OUT_PREFIX}.arpa" "${OUT_PREFIX}.klm"
echo "[Done] ${OUT_PREFIX}.klm"