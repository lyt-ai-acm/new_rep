import json
import pandas as pd

base_path = 'outputs/roberta_binary_e0/base_dev_metrics.json'
e123_path = 'outputs/roberta_binary_e0/e123_dev_metrics.json'
out_csv   = 'outputs/roberta_binary_e0/dev_compare_all.csv'

base = json.load(open(base_path, 'r', encoding='utf-8'))
e123 = json.load(open(e123_path, 'r', encoding='utf-8'))

rows = []
for k, v in base.items():
    rows.append([k, v['accuracy'], v['f1'], v['precision'], v['recall']])
for k, v in e123.items():
    rows.append([k, v['accuracy'], v['f1'], v['precision'], v['recall']])

df = pd.DataFrame(rows, columns=['Setting', 'Accuracy', 'F1', 'Precision', 'Recall']).sort_values('F1', ascending=False)
df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print(df.to_string(index=False))
print(f"[Done] {out_csv}")
