#!/opt/conda/bin/python
"""Create an audit-only RetailRocket stream with action one-hot attributes."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "data/processed/ctdg/retailrocket/events.csv"
outdir = ROOT / "data/processed/ctdg/retailrocket_oracle"
outdir.mkdir(parents=True, exist_ok=True)
frame = pd.read_csv(source)
for index, action in enumerate(("view", "addtocart", "transaction")):
    frame[f"feat_{index}"] = (frame["action_type"] == action).astype("float32")
frame.to_csv(outdir / "events.csv", index=False)
metadata = {
    "dataset": "retailrocket_oracle", "rows": len(frame),
    "source": str(source),
    "warning": "Audit-only upper bound. feat_0..2 reveal view/addtocart/transaction.",
}
(outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"wrote {outdir / 'events.csv'} ({len(frame)} rows)")
