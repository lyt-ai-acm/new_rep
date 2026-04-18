import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)

    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--orig_col", type=str, default="orig")
    p.add_argument("--cand_prefix", type=str, default="cand_")
    p.add_argument("--w_prefix", type=str, default="w_")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--top_k", type=int, default=10, help="使用前K个候选")
    p.add_argument("--alpha", type=float, default=1.0, help="权重温度幂次: w^alpha 后再归一化")
    p.add_argument("--fallback_orig", action="store_true", help="启用门控回退到原句预测")

    # 保留旧超参接口防止外层Shell报错，但在内部被软门控取代或结合使用
    p.add_argument("--w1_threshold", type=float, default=0.35)
    p.add_argument("--margin_threshold", type=float, default=0.08)

    # 创新点专属超参: 熵的边界阈值
    p.add_argument("--entropy_low", type=float, default=0.20, help="熵低于此值完全相信纠错")
    p.add_argument("--entropy_high", type=float, default=0.70, help="熵高于此值完全回退原句")
    p.add_argument("--gate_entropy_coef", type=float, default=6.0, help="门控网络中熵特征系数")
    p.add_argument("--gate_w1_coef", type=float, default=4.0, help="门控网络中w1特征系数")
    p.add_argument("--gate_margin_coef", type=float, default=4.0, help="门控网络中margin特征系数")
    p.add_argument("--gate_bias", type=float, default=-3.0, help="门控网络偏置项")
    return p.parse_args()


def predict_prob(texts, tokenizer, model, device, batch_size=64, max_len=128):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def normalize_weights(W, alpha=1.0):
    W = np.clip(W, 1e-12, None)
    if alpha != 1.0:
        W = np.power(W, alpha)
    Z = W.sum(axis=1, keepdims=True)
    Z = np.where(Z <= 0, 1.0, Z)
    return W / Z


# ==========================================
# 【创新点2】计算 N-best 分布的香农熵
# ==========================================
def compute_shannon_entropy(W):
    W_safe = np.clip(W, 1e-12, 1.0)
    H = -np.sum(W_safe * np.log(W_safe), axis=1)
    return H


def compute_dynamic_fallback_lambda(Wn, args):
    """
    轻量门控网络（单层sigmoid）:
    输入特征: 归一化熵、低w1风险、低margin风险
    输出: lambda_fallback ∈ [0, 1]
    """
    K = Wn.shape[1]
    H = compute_shannon_entropy(Wn)
    H_norm = H / np.log(K + 1e-12)

    w1 = Wn[:, 0]
    w2 = Wn[:, 1] if K >= 2 else np.zeros_like(w1)
    margin = w1 - w2

    entropy_span = max(args.entropy_high - args.entropy_low, 1e-6)
    entropy_feature = np.clip((H_norm - args.entropy_low) / entropy_span, 0.0, 1.0)
    w1_risk = np.clip(args.w1_threshold - w1, 0.0, 1.0)
    margin_risk = np.clip(args.margin_threshold - margin, 0.0, 1.0)

    z = (
        args.gate_bias
        + args.gate_entropy_coef * entropy_feature
        + args.gate_w1_coef * w1_risk
        + args.gate_margin_coef * margin_risk
    )
    lambda_fb = 1.0 / (1.0 + np.exp(-z))

    return lambda_fb, H_norm, w1, margin


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    if args.label_col not in df.columns:
        raise KeyError(f"label col '{args.label_col}' not found in {args.input_csv}")

    cand_cols_all = [c for c in df.columns if c.startswith(args.cand_prefix)]
    w_cols_all = [c for c in df.columns if c.startswith(args.w_prefix)]
    if len(cand_cols_all) == 0 or len(w_cols_all) == 0:
        raise ValueError("No candidate or weight columns found.")

    max_cand = min(len(cand_cols_all), len(w_cols_all))
    K = min(args.top_k, max_cand)

    cand_cols = [f"{args.cand_prefix}{i}" for i in range(1, K + 1)]
    w_cols = [f"{args.w_prefix}{i}" for i in range(1, K + 1)]

    y_true = df[args.label_col].astype(int).to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)

    # ==== 预测原句（用于Base/回退）====
    if args.orig_col in df.columns:
        orig_texts = df[args.orig_col].astype(str).tolist()
    elif "review" in df.columns:
        orig_texts = df["review"].astype(str).tolist()
    else:
        orig_texts = None

    p_orig = None
    if orig_texts is not None:
        p_orig = predict_prob(orig_texts, tokenizer, model, device, args.batch_size, args.max_len)

    # ==== 预测cand_1..cand_K ====
    P = []
    for c in cand_cols:
        texts = df[c].fillna("").astype(str).tolist()
        p = predict_prob(texts, tokenizer, model, device, args.batch_size, args.max_len)
        P.append(p)
    P = np.stack(P, axis=1)  # [N, K]

    # ==== 权重处理 ====
    W = df[w_cols].fillna(0.0).astype(float).to_numpy()  # [N, K]
    Wn = normalize_weights(W, alpha=args.alpha)

    p_e1 = P[:, 0]
    y_e1 = (p_e1 >= 0.5).astype(int)

    p_e2 = P.mean(axis=1)
    y_e2 = (p_e2 >= 0.5).astype(int)

    p_e3 = (P * Wn).sum(axis=1)
    y_e3 = (p_e3 >= 0.5).astype(int)

    result = {}
    if p_orig is not None:
        y_base = (p_orig >= 0.5).astype(int)
        result["Base_orig"] = compute_metrics(y_true, y_base)

    result["E1_top1"] = compute_metrics(y_true, y_e1)
    result["E2_topk_avg"] = compute_metrics(y_true, y_e2)
    result["E3_topk_weighted"] = compute_metrics(y_true, y_e3)

    # ==== 【核心创新】E4: 熵驱动的柔性自适应门控 (Entropy-Driven Soft Gating) ====
    if args.fallback_orig:
        if p_orig is None:
            raise ValueError("fallback_orig=True but no orig/review column found in input csv.")

        lambda_fb, H_norm, w1, margin = compute_dynamic_fallback_lambda(Wn, args)

        # 平滑插值 (Soft Blending): lambda * 原始概率 + (1 - lambda) * 纠错融合概率
        p_e4_dynamic = lambda_fb * p_orig + (1.0 - lambda_fb) * p_e3
        y_e4_dynamic = (p_e4_dynamic >= 0.5).astype(int)

        m_dynamic = compute_metrics(y_true, y_e4_dynamic)
        m_dynamic["mean_normalized_entropy"] = float(H_norm.mean())
        m_dynamic["mean_lambda_fallback"] = float(lambda_fb.mean())
        m_dynamic["mean_w1"] = float(w1.mean())
        m_dynamic["mean_margin"] = float(margin.mean())

        # 为了兼顾原有的日志提取逻辑，命名中保留 "_fallback" 字样
        result["E3_topk_weighted_fallback"] = m_dynamic
        result["E4_Entropy_Dynamic_Gating"] = m_dynamic  # 新增显式标签

    # 记录配置
    result["_config"] = {
        "top_k": K,
        "alpha": args.alpha,
        "fallback_orig": bool(args.fallback_orig),
        "entropy_low": args.entropy_low,
        "entropy_high": args.entropy_high,
        "gate_entropy_coef": args.gate_entropy_coef,
        "gate_w1_coef": args.gate_w1_coef,
        "gate_margin_coef": args.gate_margin_coef,
        "gate_bias": args.gate_bias,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[Done] {args.out_json}")


if __name__ == "__main__":
    main()

#
# import os
# import json
# import argparse
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
#
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--model_dir", type=str, required=True)
#     p.add_argument("--input_csv", type=str, required=True)
#     p.add_argument("--out_json", type=str, required=True)
#
#     p.add_argument("--label_col", type=str, default="label")
#     p.add_argument("--orig_col", type=str, default="orig")
#     p.add_argument("--cand_prefix", type=str, default="cand_")
#     p.add_argument("--w_prefix", type=str, default="w_")
#     p.add_argument("--max_len", type=int, default=128)
#     p.add_argument("--batch_size", type=int, default=64)
#
#     # === 创新参数 ===
#     p.add_argument("--top_k", type=int, default=10, help="使用前K个候选")
#     p.add_argument("--alpha", type=float, default=1.0, help="权重温度幂次: w^alpha 后再归一化")
#     p.add_argument("--fallback_orig", action="store_true", help="启用门控回退到原句预测")
#     p.add_argument("--w1_threshold", type=float, default=0.35, help="若w1 < 阈值则回退")
#     p.add_argument("--margin_threshold", type=float, default=0.08, help="若(w1-w2) < 阈值则回退")
#     return p.parse_args()
#
#
# def predict_prob(texts, tokenizer, model, device, batch_size=64, max_len=128):
#     probs = []
#     model.eval()
#     with torch.no_grad():
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             enc = tokenizer(
#                 batch,
#                 padding=True,
#                 truncation=True,
#                 max_length=max_len,
#                 return_tensors="pt"
#             ).to(device)
#             logits = model(**enc).logits
#             p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
#             probs.extend(p.tolist())
#     return np.array(probs)
#
#
# def compute_metrics(y_true, y_pred):
#     return {
#         "accuracy": float(accuracy_score(y_true, y_pred)),
#         "f1": float(f1_score(y_true, y_pred)),
#         "precision": float(precision_score(y_true, y_pred)),
#         "recall": float(recall_score(y_true, y_pred)),
#         "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
#     }
#
#
# def normalize_weights(W, alpha=1.0):
#     W = np.clip(W, 1e-12, None)  # 防止0导致数值问题
#     if alpha != 1.0:
#         W = np.power(W, alpha)
#     Z = W.sum(axis=1, keepdims=True)
#     Z = np.where(Z <= 0, 1.0, Z)
#     return W / Z
#
#
# def main():
#     args = parse_args()
#     os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
#
#     df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
#     if args.label_col not in df.columns:
#         raise KeyError(f"label col '{args.label_col}' not found in {args.input_csv}")
#
#     # 自动探测可用候选数
#     cand_cols_all = [c for c in df.columns if c.startswith(args.cand_prefix)]
#     w_cols_all = [c for c in df.columns if c.startswith(args.w_prefix)]
#     if len(cand_cols_all) == 0 or len(w_cols_all) == 0:
#         raise ValueError("No candidate or weight columns found.")
#
#     max_cand = min(len(cand_cols_all), len(w_cols_all))
#     K = min(args.top_k, max_cand)
#
#     cand_cols = [f"{args.cand_prefix}{i}" for i in range(1, K + 1)]
#     w_cols = [f"{args.w_prefix}{i}" for i in range(1, K + 1)]
#
#     for c in cand_cols + w_cols:
#         if c not in df.columns:
#             raise KeyError(f"Missing required column: {c}")
#
#     y_true = df[args.label_col].astype(int).to_numpy()
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
#     model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)
#
#     # ==== 预测原句（用于Base/回退）====
#     if args.orig_col in df.columns:
#         orig_texts = df[args.orig_col].astype(str).tolist()
#     elif "review" in df.columns:
#         orig_texts = df["review"].astype(str).tolist()
#     else:
#         orig_texts = None
#
#     p_orig = None
#     if orig_texts is not None:
#         p_orig = predict_prob(orig_texts, tokenizer, model, device, args.batch_size, args.max_len)
#
#     # ==== 预测cand_1..cand_K ====
#     P = []
#     for c in cand_cols:
#         texts = df[c].fillna("").astype(str).tolist()
#         p = predict_prob(texts, tokenizer, model, device, args.batch_size, args.max_len)
#         P.append(p)
#     P = np.stack(P, axis=1)  # [N, K]
#
#     # ==== 权重处理 ====
#     W = df[w_cols].fillna(0.0).astype(float).to_numpy()  # [N, K]
#     Wn = normalize_weights(W, alpha=args.alpha)
#
#     # E1: Top1
#     p_e1 = P[:, 0]
#     y_e1 = (p_e1 >= 0.5).astype(int)
#
#     # E2: 均值
#     p_e2 = P.mean(axis=1)
#     y_e2 = (p_e2 >= 0.5).astype(int)
#
#     # E3: 加权（alpha重标定后）
#     p_e3 = (P * Wn).sum(axis=1)
#     y_e3 = (p_e3 >= 0.5).astype(int)
#
#     # Base（若可用）
#     result = {}
#
#     if p_orig is not None:
#         y_base = (p_orig >= 0.5).astype(int)
#         result["Base_orig"] = compute_metrics(y_true, y_base)
#
#     result["E1_top1"] = compute_metrics(y_true, y_e1)
#     result["E2_topk_avg"] = compute_metrics(y_true, y_e2)
#     result["E3_topk_weighted"] = compute_metrics(y_true, y_e3)
#
#     # ==== E3 + fallback ====
#     if args.fallback_orig:
#         if p_orig is None:
#             raise ValueError("fallback_orig=True but no orig/review column found in input csv.")
#
#         w1 = Wn[:, 0]
#         if K >= 2:
#             w2 = Wn[:, 1]
#         else:
#             w2 = np.zeros_like(w1)
#
#         fallback_mask = (w1 < args.w1_threshold) | ((w1 - w2) < args.margin_threshold)
#         p_e3_fb = p_e3.copy()
#         p_e3_fb[fallback_mask] = p_orig[fallback_mask]
#         y_e3_fb = (p_e3_fb >= 0.5).astype(int)
#
#         m = compute_metrics(y_true, y_e3_fb)
#         m["fallback_ratio"] = float(fallback_mask.mean())
#         m["fallback_count"] = int(fallback_mask.sum())
#         result["E3_topk_weighted_fallback"] = m
#
#     # 记录配置
#     result["_config"] = {
#         "top_k": K,
#         "alpha": args.alpha,
#         "fallback_orig": bool(args.fallback_orig),
#         "w1_threshold": args.w1_threshold,
#         "margin_threshold": args.margin_threshold
#     }
#
#     with open(args.out_json, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)
#
#     print(json.dumps(result, ensure_ascii=False, indent=2))
#     print(f"[Done] {args.out_json}")
#
#
# if __name__ == "__main__":
#     main()
