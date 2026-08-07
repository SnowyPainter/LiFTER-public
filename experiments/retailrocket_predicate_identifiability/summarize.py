#!/opt/conda/bin/python
from pathlib import Path
import json, numpy as np
HERE=Path(__file__).resolve().parent
def ms(x): x=np.asarray(x,float); return {'mean':float(x.mean()),'std':float(x.std(ddof=1))}
oracle=[json.loads(p.read_text()) for p in sorted((HERE/'outputs').glob('oracle_*.json'))]
probe=json.loads((HERE/'results/probe.json').read_text())
base=json.loads((HERE.parent/'retailrocket_predicate_validation/results/summary.json').read_text())['1']
result={'k1_baseline':base,'oracle_k3':{key:ms([r['metrics'][key] for r in oracle]) for key in ('historical_negative_auc','historical_negative_ap','historical_negative_pairwise_accuracy')},'context_probe':{key:ms([r[key] for r in probe]) for key in ('balanced_accuracy','nmi','ari')}}
(HERE/'results/summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

def pm(value):
    return f"{value['mean']:.4f} ± {value['std']:.4f}"

k1_auc=base['historical_auc']; k1_ap=base['historical_ap']; k1_pair=base['pairwise_accuracy']
oracle_auc=result['oracle_k3']['historical_negative_auc']
oracle_ap=result['oracle_k3']['historical_negative_ap']
oracle_pair=result['oracle_k3']['historical_negative_pairwise_accuracy']
report=f"""# RetailRocket predicate identifiability diagnosis

The experiment uses the same chronological 262,144-event transaction-query
task and seeds 7/17/27 as the predicate-count sweep. All historical events remain
available to both models. The oracle condition changes only fact typing: the
recorded `view`, `addtocart`, and `transaction` action is supplied as a fixed
three-predicate assignment. It is an audit upper bound, not a forecasting model.

## Forecasting with exact action predicates

| Typing | Historical AUC | Historical AP | Pairwise accuracy |
|---|---:|---:|---:|
| Single predicate (K=1) | {pm(k1_auc)} | {pm(k1_ap)} | {pm(k1_pair)} |
| Fixed true action predicate (oracle K=3) | {pm(oracle_auc)} | {pm(oracle_ap)} | {pm(oracle_pair)} |

The oracle changes mean AUC by {oracle_auc['mean']-k1_auc['mean']:+.4f}, AP by
{oracle_ap['mean']-k1_ap['mean']:+.4f}, and pairwise accuracy by
{oracle_pair['mean']-k1_pair['mean']:+.4f}. Exact action typing therefore provides
only a small forecasting gain on this task.

## Decodability from LiFTER's causal fact context

| Probe metric | Result | Chance/reference |
|---|---:|---:|
| Balanced accuracy | {pm(result['context_probe']['balanced_accuracy'])} | 0.3333 |
| NMI | {pm(result['context_probe']['nmi'])} | 0.0000 |
| ARI | {pm(result['context_probe']['ari'])} | 0.0000 |

A supervised diagnostic probe can recover the hidden current action substantially
above chance from the same strictly earlier structural context available to the
fact encoder. Thus the input contains action-correlated information. The ordinary
future-link objective nevertheless does not organize K>1 predicates around these
actions, as established by the separate K sweep and predicate-shuffle test.

## Decision

The observed collapse is partly an objective-identification issue, but it is not
the main forecasting bottleneck: an exact, semantically correct predicate supplies
only +{oracle_ap['mean']-k1_ap['mean']:.4f} historical AP. RetailRocket therefore
does not support redesigning LiFTER around action recovery merely to improve link
forecasting. A stronger predicate-learning claim requires a task in which latent
mechanisms affect the target conditionally and exact typed rules materially beat
the single-predicate program.
"""
(HERE/'results/report.md').write_text(report)
