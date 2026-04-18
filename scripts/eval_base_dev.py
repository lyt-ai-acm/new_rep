import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

dev_path = 'data/splits/dev.csv'
model_dir = 'outputs/roberta_binary_e0/best_model'
out_path  = 'outputs/roberta_binary_e0/base_dev_metrics.json'

df = pd.read_csv(dev_path, encoding='utf-8-sig')
texts = df['review'].astype(str).tolist()
y_true = df['label'].astype(int).to_numpy()

tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

bs = 64
probs = []
with torch.no_grad():
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.extend(p.tolist())

y_prob = np.array(probs)
y_pred = (y_prob >= 0.5).astype(int)

res = {
    "Base_orig": {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
}

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print(json.dumps(res, ensure_ascii=False, indent=2))
print(f"[Done] {out_path}")