from pathlib import Path
import pandas as pd

root = Path("runs/papa_dyn_ladder")
latest = sorted(root.glob("*"))[-1]
print("latest:", latest)

rows = []
for run_dir in latest.iterdir():
    f = run_dir / "resp_dyn_summary.csv"
    if f.exists():
        df = pd.read_csv(f)
        df["run"] = run_dir.name
        rows.append(df)

df = pd.concat(rows, ignore_index=True)
summary = (
    df.groupby(["run", "resp_dyn_variant"])
      [["resp_dyn_acc", "resp_dyn_bal_acc", "resp_dyn_f1_macro", "resp_dyn_f1_weighted"]]
      .mean()
      .sort_values(["run", "resp_dyn_bal_acc"], ascending=[True, False])
)

print(summary.to_string())
