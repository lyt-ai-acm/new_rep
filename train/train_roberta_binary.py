# # -*- coding: utf-8 -*-
# import os
# import json
# import argparse
# import numpy as np
# import pandas as pd
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
#
# from datasets import Dataset, DatasetDict
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     DataCollatorWithPadding,
#     Trainer,
#     TrainingArguments,
#     set_seed,
# )
#
#
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_path", type=str, required=True)
#     p.add_argument("--output_dir", type=str, required=True)
#     p.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
#     p.add_argument("--text_col", type=str, default="review")
#     p.add_argument("--label_col", type=str, default="label")
#     p.add_argument("--test_size", type=float, default=0.1)
#     p.add_argument("--dev_size", type=float, default=0.1)
#     p.add_argument("--max_len", type=int, default=128)
#     p.add_argument("--epochs", type=int, default=3)
#     p.add_argument("--batch_size", type=int, default=16)
#     p.add_argument("--lr", type=float, default=2e-5)
#     p.add_argument("--weight_decay", type=float, default=0.01)
#     p.add_argument("--warmup_ratio", type=float, default=0.06)
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--fp16", action="store_true")
#     return p.parse_args()
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds, average="binary", pos_label=1),
#         "precision": precision_score(labels, preds, average="binary", pos_label=1, zero_division=0),
#         "recall": recall_score(labels, preds, average="binary", pos_label=1, zero_division=0),
#     }
#
#
# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#     set_seed(args.seed)
#
#     df = pd.read_csv(args.data_path, encoding="utf-8-sig")
#     df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
#     df = df[df[args.text_col] != ""].copy()
#     df[args.label_col] = df[args.label_col].astype(int)
#
#     train_dev, test = train_test_split(
#         df, test_size=args.test_size, random_state=args.seed, stratify=df[args.label_col]
#     )
#     dev_ratio = args.dev_size / (1.0 - args.test_size)
#     train, dev = train_test_split(
#         train_dev, test_size=dev_ratio, random_state=args.seed, stratify=train_dev[args.label_col]
#     )
#
#     train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False, encoding="utf-8-sig")
#     dev.to_csv(os.path.join(args.output_dir, "dev.csv"), index=False, encoding="utf-8-sig")
#     test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False, encoding="utf-8-sig")
#
#     ds = DatasetDict({
#         "train": Dataset.from_pandas(train[[args.text_col, args.label_col]].rename(columns={args.text_col:"text", args.label_col:"label"}), preserve_index=False),
#         "dev": Dataset.from_pandas(dev[[args.text_col, args.label_col]].rename(columns={args.text_col:"text", args.label_col:"label"}), preserve_index=False),
#         "test": Dataset.from_pandas(test[[args.text_col, args.label_col]].rename(columns={args.text_col:"text", args.label_col:"label"}), preserve_index=False),
#     })
#
#     tok = AutoTokenizer.from_pretrained(args.model_name)
#
#     def tok_fn(batch):
#         return tok(batch["text"], truncation=True, max_length=args.max_len)
#
#     ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
#     collator = DataCollatorWithPadding(tokenizer=tok)
#
#     model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
#
#     targs = TrainingArguments(
#         output_dir=args.output_dir,
#         learning_rate=args.lr,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         num_train_epochs=args.epochs,
#         weight_decay=args.weight_decay,
#         warmup_ratio=args.warmup_ratio,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         logging_strategy="steps",
#         logging_steps=100,
#         load_best_model_at_end=True,
#         metric_for_best_model="f1",
#         greater_is_better=True,
#         fp16=args.fp16,
#         report_to="none",
#         seed=args.seed,
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=targs,
#         train_dataset=ds_tok["train"],
#         eval_dataset=ds_tok["dev"],
#         tokenizer=tok,
#         data_collator=collator,
#         compute_metrics=compute_metrics
#     )
#
#     trainer.train()
#     dev_metrics = trainer.evaluate(ds_tok["dev"])
#     test_metrics = trainer.evaluate(ds_tok["test"])
#
#     pred = trainer.predict(ds_tok["test"])
#     y_true = pred.label_ids
#     y_pred = np.argmax(pred.predictions, axis=-1)
#     cm = confusion_matrix(y_true, y_pred).tolist()
#
#     best_dir = os.path.join(args.output_dir, "best_model")
#     trainer.save_model(best_dir)
#     tok.save_pretrained(best_dir)
#
#     out = {
#         "dev": {k: float(v) for k, v in dev_metrics.items() if isinstance(v, (int, float))},
#         "test": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
#         "test_confusion_matrix": cm
#     }
#     with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(out, f, ensure_ascii=False, indent=2)
#
#     print("[Done]", args.output_dir)
#
#
# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ==========================================
# 【创新点1】监督对比损失 (SupCon Loss)
# ==========================================
def supervised_contrastive_loss(features, labels, temperature=0.07):
    device = features.device
    batch_size = features.shape[0]

    features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - sim_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mask_sum = mask.sum(1)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    loss = -mean_log_prob_pos.mean()
    return loss


class ContrastiveCELossTrainer(Trainer):
    def __init__(self, scl_weight=0.1, scl_temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scl_weight = scl_weight
        self.scl_temperature = scl_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # 强制输出 hidden_states 以提取 [CLS]
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss()
        loss_ce = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        # 提取最后一层 [CLS] 向量
        last_hidden_state = outputs.hidden_states[-1]
        cls_embeds = last_hidden_state[:, 0, :]

        loss_scl = supervised_contrastive_loss(cls_embeds, labels, temperature=self.scl_temperature)

        total_loss = loss_ce + self.scl_weight * loss_scl
        if return_outputs:
            outputs.hidden_states = None
            return total_loss, outputs
        return total_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--text_col", type=str, default="review")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--dev_size", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")

    # 新增创新参数
    p.add_argument("--scl_weight", type=float, default=0.1, help="SupCon对比损失的权重系数")
    p.add_argument("--scl_temperature", type=float, default=0.07, help="对比学习温度参数")
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", pos_label=1),
        "precision": precision_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, average="binary", pos_label=1, zero_division=0),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.data_path, encoding="utf-8-sig")
    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df = df[df[args.text_col] != ""].copy()
    df[args.label_col] = df[args.label_col].astype(int)

    train_dev, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df[args.label_col]
    )
    dev_ratio = args.dev_size / (1.0 - args.test_size)
    train, dev = train_test_split(
        train_dev, test_size=dev_ratio, random_state=args.seed, stratify=train_dev[args.label_col]
    )

    train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False, encoding="utf-8-sig")
    dev.to_csv(os.path.join(args.output_dir, "dev.csv"), index=False, encoding="utf-8-sig")
    test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False, encoding="utf-8-sig")

    ds = DatasetDict({
        "train": Dataset.from_pandas(
            train[[args.text_col, args.label_col]].rename(columns={args.text_col: "text", args.label_col: "label"}),
            preserve_index=False),
        "dev": Dataset.from_pandas(
            dev[[args.text_col, args.label_col]].rename(columns={args.text_col: "text", args.label_col: "label"}),
            preserve_index=False),
        "test": Dataset.from_pandas(
            test[[args.text_col, args.label_col]].rename(columns={args.text_col: "text", args.label_col: "label"}),
            preserve_index=False),
    })

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
    )

    # 替换为带对比学习的 Trainer
    trainer = ContrastiveCELossTrainer(
        scl_weight=args.scl_weight,
        scl_temperature=args.scl_temperature,
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["dev"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    dev_metrics = trainer.evaluate(ds_tok["dev"])
    test_metrics = trainer.evaluate(ds_tok["test"])

    pred = trainer.predict(ds_tok["test"])
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)
    cm = confusion_matrix(y_true, y_pred).tolist()

    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)

    out = {
        "dev": {k: float(v) for k, v in dev_metrics.items() if isinstance(v, (int, float))},
        "test": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        "test_confusion_matrix": cm
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[Done]", args.output_dir)


if __name__ == "__main__":
    main()
