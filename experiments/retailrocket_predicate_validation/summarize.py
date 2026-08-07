#!/opt/conda/bin/python
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def ms(values):
    x=np.asarray(values,float); return {"mean":float(x.mean()),"std":float(x.std(ddof=1))}
def main():
    rows=[json.loads(p.read_text()) for p in sorted((HERE/"outputs").glob("*.json"))]
    summary={}
    for k in (1,2,3,4,8):
        selected=[r for r in rows if r["k"]==k]
        if not selected: continue
        summary[str(k)]={
            "historical_auc":ms([r["normal"]["historical_negative_auc"] for r in selected]),
            "historical_ap":ms([r["normal"]["historical_negative_ap"] for r in selected]),
            "pairwise_accuracy":ms([r["normal"]["historical_negative_pairwise_accuracy"] for r in selected]),
            "global_shuffle_auc_drop":ms([r["normal"]["historical_negative_auc"]-r["global_predicate_shuffle"]["historical_negative_auc"] for r in selected]),
            "global_shuffle_ap_drop":ms([r["normal"]["historical_negative_ap"]-r["global_predicate_shuffle"]["historical_negative_ap"] for r in selected]),
            "action_nmi":ms([r["action_recovery"]["nmi"] for r in selected]),
            "action_ari":ms([r["action_recovery"]["ari"] for r in selected]),
            "action_macro_recall":ms([r["action_recovery"]["permutation_matched_macro_recall"] for r in selected]),
        }
    destination=HERE/"results"/"summary.json"; destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(summary,indent=2)+"\n"); print(f"saved {destination}")
if __name__=="__main__": main()
