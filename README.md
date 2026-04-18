# Weibo Homophone Normalization + KenLM Rerank + Sentiment

## 目标
- 词级谐音归一化（Top10 n-best）
- KenLM重排 + 过纠抑制
- 微博情感二分类（先跑通），后续扩展三分类

## 数据
- `data/Weibo_senti_100k.csv`，列：`label,review`
- 标签：`1=正面`，`0=负面`
- 读取编码建议：`utf-8-sig`

## 快速开始

### 1. 切分数据
```bash
python scripts/01_split_weibo.py --input_csv data/Weibo_senti_100k.csv --out_dir data/splits
```

### 2. 准备LM语料（jieba）
```bash
python scripts/02_tokenize_for_lm_jieba.py --input_csv data/splits/train.csv --output_txt data/lm/corpus_jieba.txt
```

### 3. 训练KenLM（WSL/Linux）
```bash
bash scripts/03_train_kenlm.sh data/lm/corpus_jieba.txt models/lm_jieba_5gram
```

### 4. 生成Top10归一化候选
```bash
python pipeline/run_jieba.py \
  --input_csv data/splits/dev.csv \
  --output_csv outputs/norm/dev_top10_jieba.csv \
  --kenlm_path models/lm_jieba_5gram.klm \
  --word_homo resources/chinese_homophone_word.txt \
  --char_homo resources/chinese_homophone_char.txt
```

### 5. 训练二分类模型（E0）
```bash
python train/train_roberta_binary.py \
  --data_path data/Weibo_senti_100k.csv \
  --output_dir outputs/roberta_binary_e0 \
  --model_name hfl/chinese-roberta-wwm-ext
```

### 6. 评测E1/E2/E3
```bash
python train/infer_with_nbest.py \
  --model_dir outputs/roberta_binary_e0/best_model \
  --input_csv outputs/norm/dev_top10_jieba.csv \
  --out_json outputs/roberta_binary_e0/e123_dev_metrics.json
```

### 7. 熵驱动动态回退（E4）
```bash
python train/infer_with_nbest.py \
  --model_dir outputs/roberta_binary_e0/best_model \
  --input_csv outputs/norm/dev_top10_jieba.csv \
  --out_json outputs/roberta_binary_e0/e4_entropy_metrics.json \
  --fallback_orig \
  --entropy_low 0.20 \
  --entropy_high 0.70
```
