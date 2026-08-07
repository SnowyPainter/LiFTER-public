#!/opt/conda/bin/python
import argparse,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(); p.add_argument('--scale',default='pilot'); a=p.parse_args(); records=[]
 for f in (HERE/'outputs').glob(f'*_{a.scale}.json'): records.append(json.loads(f.read_text()))
 summary={}
 for d in sorted({r['dataset'] for r in records}):
  summary[d]={}
  for v in sorted({r['variant'] for r in records if r['dataset']==d}):
   rows=[r for r in records if r['dataset']==d and r['variant']==v]
   summary[d][v]={k:{"mean":float(np.mean([r[k] for r in rows])),"std":float(np.std([r[k] for r in rows],ddof=1))} for k in ('historical_auc','historical_ap','random_auc','random_ap')}
 out=HERE/'results'/f'{a.scale}_summary.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,indent=2)+'\n'); print(out)
if __name__=='__main__':main()
