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

## 批量骨干对比实验（裸版 vs 软融合/对比学习增强）

```bash
bash scripts/run_experiments.sh \
  --data_path data/Weibo_senti_100k.csv \
  --nbest_csv outputs/norm/dev_top10_jieba.csv \
  --output_dir outputs/experiments \
  --epochs 3 --batch_size 16 --seed 42
```

- 裸版训练：`train/train_baseline.py`（仅交叉熵）
- 增强版训练：`train/train_contrastive.py`（交叉熵 + SupCon）
- 增强版融合评测：`train/infer_nbest_entropy.py`
- 汇总输出：`outputs/experiments/metrics.json`
