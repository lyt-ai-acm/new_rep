import sys
print("Python:", sys.executable, sys.version)

mods = [
    "numpy","pandas","sklearn","scipy","tqdm",
    "jieba","pypinyin","kenlm",
    "torch","transformers","datasets","accelerate","sentencepiece","evaluate",
    "matplotlib","seaborn","yaml","regex","fsspec"
]
for m in mods:
    try:
        __import__(m)
        print("[OK]", m)
    except Exception as e:
        print("[ERR]", m, "->", e)

# torch info
try:
    import torch
    print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("torch check failed:", e)