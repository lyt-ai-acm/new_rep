#!/usr/bin/env bash
set -euo pipefail

# =========================
# 0) 可改参数区
# =========================
PROJECT_DIR="/root/new_model"
cd "$PROJECT_DIR"

# 初始checkpoint（你当前最好的）
BASE_CKPT="outputs/roberta_binary_e0/best_model"

# 训练/验证数据
TRAIN_CSV="data/splits/train.csv"
DEV_CSV="data/splits/dev.csv"

# 你的训练脚本（如参数名不同，按你脚本改）
TRAIN_PY="train/train_roberta.py"

# 评测脚本
INFER_PY="train/infer_with_nbest.py"

# nbest评测集（存在才跑）
NBEST_20K="outputs/norm/dev_top10_jieba_3g20k_strict_labeled.csv"
NBEST_40P="outputs/norm/dev_top10_jieba_3g40p_labeled.csv"

# 网格：续训轮数 + 学习率
EPOCHS_LIST=("1" "2" "3")
LR_LIST=("1e-5" "5e-6" "2e-6")
SEED_LIST=("42" "3407")

# 评测固定参数
TOPK=5
ALPHA=2.0
W1=0.45
MARGIN=0.12
LAM_LIST="0.6,0.7,0.8,0.9"

# 输出目录
RUN_ROOT="outputs/resume_grid"
mkdir -p "$RUN_ROOT"

# =========================
# 1) 工具函数
# =========================
run_train () {
  local out_dir="$1"
  local lr="$2"
  local ep="$3"
  local seed="$4"

  mkdir -p "$out_dir"

  echo ">>> [TRAIN] out=$out_dir lr=$lr ep=$ep seed=$seed"

  # ===== 按你train_roberta.py参数改这里 =====
  PYTHONPATH=. python "$TRAIN_PY" \
    --train_csv "$TRAIN_CSV" \
    --dev_csv "$DEV_CSV" \
    --model_name_or_path "$BASE_CKPT" \
    --output_dir "$out_dir" \
    --learning_rate "$lr" \
    --num_train_epochs "$ep" \
    --seed "$seed" \
    --do_train \
    --do_eval
  # =======================================
}

run_eval_one_nbest () {
  local model_dir="$1"
  local nbest_csv="$2"
  local tag="$3"
  local out_json="$4"

  echo ">>> [EVAL] model=$model_dir nbest=$tag"
  PYTHONPATH=. python "$INFER_PY" \
    --model_dir "$model_dir" \
    --input_csv "$nbest_csv" \
    --out_json "$out_json" \
    --top_k "$TOPK" \
    --alpha "$ALPHA" \
    --fallback_orig \
    --w1_threshold "$W1" \
    --margin_threshold "$MARGIN" \
    --mix_lambda_list "$LAM_LIST"
}

# 从json提取关键分数（Base / E3_fb / E4_best）
extract_scores () {
  local json_file="$1"
  python - << 'PY' "$json_file"
import json,sys
p=sys.argv[1]
x=json.load(open(p,'r',encoding='utf-8'))
base=x.get("Base_orig",{}).get("f1",-1)
e3fb=x.get("E3_topk_weighted_fallback",{}).get("f1",-1)
e4b=x.get("E4_best",{}).get("f1",-1)
e4s=x.get("E4_best",{}).get("setting","")
print(f"{base}\t{e3fb}\t{e4b}\t{e4s}")
PY
}

# =========================
# 2) 主循环：训练 + 评测
# =========================
SUMMARY_TSV="$RUN_ROOT/summary.tsv"
echo -e "run_id\tlr\tepochs\tseed\tnbest\tbase_f1\te3fb_f1\te4best_f1\te4best_setting\tjson" > "$SUMMARY_TSV"

for ep in "${EPOCHS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      RUN_ID="resume_ep${ep}_lr${lr}_s${seed}"
      OUT_DIR="$RUN_ROOT/$RUN_ID"

      run_train "$OUT_DIR" "$lr" "$ep" "$seed"

      # 兼容best_model目录或直接output_dir
      MODEL_DIR="$OUT_DIR/best_model"
      if [ ! -d "$MODEL_DIR" ]; then
        MODEL_DIR="$OUT_DIR"
      fi

      # 20k
      if [ -f "$NBEST_20K" ]; then
        J20="$OUT_DIR/eval_3g20k.json"
        run_eval_one_nbest "$MODEL_DIR" "$NBEST_20K" "3g20k" "$J20"
        read -r b e3 e4 e4s <<< "$(extract_scores "$J20")"
        echo -e "${RUN_ID}\t${lr}\t${ep}\t${seed}\t3g20k\t${b}\t${e3}\t${e4}\t${e4s}\t${J20}" >> "$SUMMARY_TSV"
      fi

      # 40p
      if [ -f "$NBEST_40P" ]; then
        J40="$OUT_DIR/eval_3g40p.json"
        run_eval_one_nbest "$MODEL_DIR" "$NBEST_40P" "3g40p" "$J40"
        read -r b e3 e4 e4s <<< "$(extract_scores "$J40")"
        echo -e "${RUN_ID}\t${lr}\t${ep}\t${seed}\tn3g40p\t${b}\t${e3}\t${e4}\t${e4s}\t${J40}" >> "$SUMMARY_TSV"
      fi

    done
  done
done

# =========================
# 3) 自动选最佳checkpoint（按E4_best_f1）
# =========================
BEST_TXT="$RUN_ROOT/best_by_e4.txt"
python - << 'PY' "$SUMMARY_TSV" "$BEST_TXT"
import pandas as pd,sys
tsv,best_txt=sys.argv[1],sys.argv[2]
df=pd.read_csv(tsv,sep='\t')
if len(df)==0:
    open(best_txt,'w').write("No rows.\n")
    print("No rows.")
    raise SystemExit(0)

df['e4best_f1']=pd.to_numeric(df['e4best_f1'],errors='coerce')
best=df.sort_values(['e4best_f1','base_f1'],ascending=False).iloc[0]
msg=[]
msg.append("Best by e4best_f1:")
for k in ['run_id','lr','epochs','seed','nbest','base_f1','e3fb_f1','e4best_f1','e4best_setting','json']:
    msg.append(f"{k}: {best[k]}")
open(best_txt,'w',encoding='utf-8').write("\n".join(msg)+"\n")
print("\n".join(msg))
PY

echo ">>> Done."
echo "Summary: $SUMMARY_TSV"
echo "Best:    $BEST_TXT"