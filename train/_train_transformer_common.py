# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common_senti_train import compute_metrics, load_and_split_data, save_metrics, set_seed


def build_parser(default_model_name: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default=default_model_name)
    ap.add_argument("--text_col", type=str, default="review")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--dev_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    return ap


def run_transformer_finetune(args):
    set_seed(args.seed)
    train_df, dev_df, test_df, _, id2label = load_and_split_data(
        data_path=args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
        dev_size=args.dev_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    num_labels = len(id2label)

    def to_hf(df):
        return Dataset.from_pandas(
            df[[args.text_col, "label_id"]].rename(columns={args.text_col: "text", "label_id": "labels"}),
            preserve_index=False,
        )

    ds = DatasetDict({"train": to_hf(train_df), "dev": to_hf(dev_df), "test": to_hf(test_df)})

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        trust_remote_code=args.trust_remote_code,
    )

    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        m = compute_metrics(labels, preds, num_labels)
        return {k: v for k, v in m.items() if k != "confusion_matrix"}

    os.makedirs(args.output_dir, exist_ok=True)
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

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["dev"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=_compute,
    )

    trainer.train()
    pred_dev = trainer.predict(ds_tok["dev"])
    pred_test = trainer.predict(ds_tok["test"])
    dev_metrics = compute_metrics(pred_dev.label_ids, np.argmax(pred_dev.predictions, axis=-1), num_labels=num_labels)
    test_metrics = compute_metrics(pred_test.label_ids, np.argmax(pred_test.predictions, axis=-1), num_labels=num_labels)

    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    save_metrics(args.output_dir, dev_metrics, test_metrics, id2label, vars(args))
    print("[Done]", args.output_dir)
