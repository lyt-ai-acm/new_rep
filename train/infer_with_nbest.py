import os
import json
import argparse
import importlib
import importlib.util
import inspect
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


def detect_model_backend(model_dir):
    if os.path.exists(os.path.join(model_dir, "model_meta.json")):
        return "classical"
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return "hf"
    raise ValueError(
        f"Unrecognized model in {model_dir}. Expected either model_meta.json (classical) or config.json (HuggingFace)."
    )


def _load_generic_training_module():
    try:
        return importlib.import_module("_generic_training")
    except ImportError:
        pass

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(repo_root, "_generic_training.py"),
        os.path.join(repo_root, "train", "_generic_training.py"),
        os.path.join(repo_root, "scripts", "_generic_training.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("_generic_training", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader, f"Failed to load module spec from {path}"
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("Could not locate _generic_training.py required for classical model inference.")


def _vocab_size(vocab):
    if hasattr(vocab, "__len__"):
        try:
            return len(vocab)
        except Exception:
            pass
    for attr in ("stoi", "token2idx", "char2idx", "vocab"):
        obj = getattr(vocab, attr, None)
        if isinstance(obj, dict):
            return len(obj)
    return None


def _pad_id(vocab):
    for attr in ("pad_id", "pad_idx"):
        if hasattr(vocab, attr):
            return int(getattr(vocab, attr))
    for attr in ("stoi", "token2idx", "char2idx", "vocab"):
        d = getattr(vocab, attr, None)
        if isinstance(d, dict):
            for key in ("[PAD]", "<pad>", "<PAD>", "PAD"):
                if key in d:
                    return int(d[key])
    return 0


def _unk_id(vocab):
    for attr in ("unk_id", "unk_idx"):
        if hasattr(vocab, attr):
            return int(getattr(vocab, attr))
    for attr in ("stoi", "token2idx", "char2idx", "vocab"):
        d = getattr(vocab, attr, None)
        if isinstance(d, dict):
            for key in ("[UNK]", "<unk>", "<UNK>", "UNK"):
                if key in d:
                    return int(d[key])
    return 1


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]
    raise TypeError("Unsupported model output type for logits extraction.")


def _load_char_vocab(char_vocab_cls, vocab_path):
    if hasattr(char_vocab_cls, "from_json"):
        return char_vocab_cls.from_json(vocab_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for candidate in (data, data.get("stoi"), data.get("token2idx"), data.get("char2idx"), data.get("vocab")):
        if candidate is None:
            continue
        try:
            return char_vocab_cls(candidate)
        except Exception:
            continue
    raise ValueError(f"Failed to initialize CharVocab from {vocab_path}.")


def _build_classical_model(model_cls, vocab, meta, device):
    sig = inspect.signature(model_cls.__init__)
    kwargs = {}
    vsize = _vocab_size(vocab)
    num_classes = int(meta.get("num_classes", meta.get("num_labels", 2)))
    defaults = {
        "vocab": vocab,
        "char_vocab": vocab,
        "vocab_size": vsize,
        "num_classes": num_classes,
        "num_labels": num_classes,
        "pad_idx": _pad_id(vocab),
        "pad_id": _pad_id(vocab),
        "embed_dim": int(meta.get("embed_dim", meta.get("embedding_dim", 128))),
        "embedding_dim": int(meta.get("embedding_dim", meta.get("embed_dim", 128))),
        "hidden_dim": int(meta.get("hidden_dim", meta.get("lstm_hidden_dim", 128))),
        "dropout": float(meta.get("dropout", 0.2)),
        "kernel_sizes": meta.get("kernel_sizes", [3, 4, 5]),
        "num_filters": int(meta.get("num_filters", 100)),
    }

    for name, p in list(sig.parameters.items())[1:]:
        if name in meta:
            kwargs[name] = meta[name]
        elif name in defaults and defaults[name] is not None:
            kwargs[name] = defaults[name]
        elif p.default is inspect.Parameter.empty and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(f"Missing required init argument '{name}' for {model_cls.__name__}.")
    model = model_cls(**kwargs).to(device)
    return model


def _load_classical_model(model_dir, device):
    module = _load_generic_training_module()
    if not all(hasattr(module, n) for n in ("CharVocab", "TextCNNClassifier", "BiLSTMAttnClassifier")):
        raise AttributeError("_generic_training.py must define CharVocab, TextCNNClassifier, BiLSTMAttnClassifier.")

    with open(os.path.join(model_dir, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_type = str(meta.get("model_type", "")).lower()
    vocab = _load_char_vocab(module.CharVocab, os.path.join(model_dir, "vocab.json"))

    if "textcnn" in model_type:
        model_cls = module.TextCNNClassifier
    elif "bilstm" in model_type:
        model_cls = module.BiLSTMAttnClassifier
    else:
        raise ValueError(f"Unsupported classical model_type '{model_type}' in model_meta.json")

    model = _build_classical_model(model_cls, vocab, meta, device)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    if isinstance(state, torch.nn.Module):
        model = state.to(device)
    else:
        sd = state.get("state_dict", state.get("model_state_dict", state)) if isinstance(state, dict) else state
        load_info = model.load_state_dict(sd, strict=False)
        missing = list(getattr(load_info, "missing_keys", []))
        unexpected = list(getattr(load_info, "unexpected_keys", []))
        if missing or unexpected:
            print(f"[Warn] model.pt key mismatch. missing={missing} unexpected={unexpected}")
    model.eval()
    return model, vocab


def _encode_text_with_vocab(text, vocab, max_len):
    for method in ("encode", "encode_text", "text_to_ids", "numericalize", "transform"):
        fn = getattr(vocab, method, None)
        if fn is None:
            continue
        for kwargs in ({"max_len": max_len}, {"max_length": max_len}, {}):
            try:
                ids = fn(text, **kwargs) if kwargs else fn(text)
                if isinstance(ids, tuple):
                    ids = ids[0]
                if ids is not None:
                    ids = list(ids)
                    break
            except TypeError:
                continue
        else:
            ids = None
        if ids is not None:
            break
    else:
        mapping = None
        for attr in ("stoi", "token2idx", "char2idx", "vocab"):
            d = getattr(vocab, attr, None)
            if isinstance(d, dict):
                mapping = d
                break
        if mapping is None:
            raise ValueError("Cannot encode text: CharVocab has no supported encode method or token map.")
        unk = _unk_id(vocab)
        ids = [int(mapping.get(ch, unk)) for ch in str(text)]

    ids = ids[:max_len]
    pad = _pad_id(vocab)
    if len(ids) < max_len:
        ids = ids + [pad] * (max_len - len(ids))
    return ids


def predict_prob_classical(texts, vocab, model, device, batch_size=64, max_len=128):
    probs = []
    model.eval()
    pad = _pad_id(vocab)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            ids = [_encode_text_with_vocab(t, vocab, max_len) for t in batch]
            x = torch.tensor(ids, dtype=torch.long, device=device)
            lengths = (x != pad).sum(dim=1)
            try:
                outputs = model(x, lengths)
            except TypeError:
                outputs = model(x)
            logits = _extract_logits(outputs)
            if logits.shape[-1] == 1:
                p = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
            else:
                p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


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
    backend = detect_model_backend(args.model_dir)
    if backend == "hf":
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)
        predictor = lambda texts: predict_prob(texts, tokenizer, model, device, args.batch_size, args.max_len)
    else:
        model, vocab = _load_classical_model(args.model_dir, device)
        predictor = lambda texts: predict_prob_classical(texts, vocab, model, device, args.batch_size, args.max_len)

    # ==== 预测原句（用于Base/回退）====
    if args.orig_col in df.columns:
        orig_texts = df[args.orig_col].astype(str).tolist()
    elif "review" in df.columns:
        orig_texts = df["review"].astype(str).tolist()
    else:
        orig_texts = None

    p_orig = None
    if orig_texts is not None:
        p_orig = predictor(orig_texts)

    # ==== 预测cand_1..cand_K ====
    P = []
    for c in cand_cols:
        texts = df[c].fillna("").astype(str).tolist()
        p = predictor(texts)
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

        # 计算归一化香农熵 H_norm ∈ [0, 1]
        H = compute_shannon_entropy(Wn)
        H_norm = H / np.log(K + 1e-12)

        # 计算连续插值系数 lambda_fallback
        lambda_fb = np.clip((H_norm - args.entropy_low) / (args.entropy_high - args.entropy_low), 0.0, 1.0)

        # 针对极度不确定的单点(w1 < threshold)依然保持兜底拦截，防止极端情况
        w1 = Wn[:, 0]
        hard_fallback_mask = (w1 < args.w1_threshold)
        lambda_fb[hard_fallback_mask] = 1.0

        # 平滑插值 (Soft Blending): lambda * 原始概率 + (1 - lambda) * 纠错融合概率
        p_e4_dynamic = lambda_fb * p_orig + (1.0 - lambda_fb) * p_e3
        y_e4_dynamic = (p_e4_dynamic >= 0.5).astype(int)

        m_dynamic = compute_metrics(y_true, y_e4_dynamic)
        m_dynamic["mean_normalized_entropy"] = float(H_norm.mean())
        m_dynamic["mean_lambda_fallback"] = float(lambda_fb.mean())

        # 为了兼顾原有的日志提取逻辑，命名中保留 "_fallback" 字样
        result["E3_topk_weighted_fallback"] = m_dynamic
        result["E4_Entropy_Dynamic_Gating"] = m_dynamic  # 新增显式标签

    # 记录配置
    result["_config"] = {
        "top_k": K,
        "alpha": args.alpha,
        "fallback_orig": bool(args.fallback_orig),
        "entropy_low": args.entropy_low,
        "entropy_high": args.entropy_high
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
