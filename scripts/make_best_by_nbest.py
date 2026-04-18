#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

summary_path = Path("/root/autodl-tmp/outputs/resume_grid/summary.tsv")
out_csv = Path("/root/autodl-tmp/outputs/resume_grid/best_by_nbest.csv")
out_md = Path("/root/autodl-tmp/outputs/resume_grid/best_by_nbest.md")

if not summary_path.exists():
    raise FileNotFoundError(f"Not found: {summary_path}")

df = pd.read_csv(summary_path, sep="\t")

# 数值列转为float
for c in ["base_f1", "e3fb_f1", "e4best_f1", "lr", "epochs", "seed"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 提升值
df["delta_f1"] = df["e4best_f1"] - df["base_f1"]

# 每个nbest选最优：先按e4best_f1降序，再按delta_f1降序，再按base_f1降序
best = (
    df.sort_values(["nbest", "e4best_f1", "delta_f1", "base_f1"], ascending=[True, False, False, False])
      .groupby("nbest", as_index=False)
      .head(1)
      .copy()
)

# 输出列（你可以按需增减）
cols = [
    "nbest", "run_id", "lr", "epochs", "seed",
    "base_f1", "e3fb_f1", "e4best_f1", "delta_f1",
    "e4best_setting", "model_dir", "json"
]
best = best[cols]

# 保留显示精度
for c in ["base_f1", "e3fb_f1", "e4best_f1", "delta_f1"]:
    best[c] = best[c].map(lambda x: f"{x:.10f}")

# 写CSV
best.to_csv(out_csv, index=False, encoding="utf-8")

# 写Markdown
md_lines = []
md_lines.append("# Best config by nbest\n")
md_lines.append("| nbest | run_id | lr | epochs | seed | base_f1 | e3fb_f1 | e4best_f1 | ΔF1 (e4-base) | e4best_setting |")
md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
for _, r in best.iterrows():
    md_lines.append(
        f"| {r['nbest']} | {r['run_id']} | {r['lr']} | {int(float(r['epochs']))} | {int(float(r['seed']))} | "
        f"{r['base_f1']} | {r['e3fb_f1']} | {r['e4best_f1']} | {r['delta_f1']} | {r['e4best_setting']} |"
    )

out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print(f"[OK] CSV: {out_csv}")
print(f"[OK] MD : {out_md}")
print("\nPreview:")
print(best.to_string(index=False))