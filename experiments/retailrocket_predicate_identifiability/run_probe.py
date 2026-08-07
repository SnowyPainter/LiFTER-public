#!/opt/conda/bin/python
"""Can LiFTER's strictly causal fact input decode the current action type?"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import balanced_accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from evaluation.evaluate import build_lifter_causal_fact_context, read_csv_tail
from models.lifter import LinkFactEncoder

class Probe(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=LinkFactEncoder(0,8,64,0.0); self.head=nn.Linear(64,3)
    def forward(self,x): return self.head(self.encoder(x))

def main():
    frame=read_csv_tail(ROOT/'data/processed/ctdg/retailrocket/events.csv',262144).sort_values('timestamp',kind='stable').reset_index(drop=True)
    context=build_lifter_causal_fact_context(frame,8)
    x=np.concatenate((np.ones((len(frame),1),np.float32),context.reshape(len(frame),-1)),axis=1)
    y=frame['action_type'].map({'view':0,'addtocart':1,'transaction':2}).to_numpy(np.int64)
    split=int(len(frame)*.85); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xt=torch.tensor(x[:split]); yt=torch.tensor(y[:split]); xe=torch.tensor(x[split:],device=device); ye=y[split:]
    counts=np.bincount(y[:split],minlength=3); weights=torch.tensor(len(y[:split])/(3*counts),dtype=torch.float32,device=device)
    records=[]
    for seed in (7,17,27):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model=Probe().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(seed)
        for _ in range(10):
            order=torch.randperm(split,generator=generator)
            model.train()
            for start in range(0,split,1024):
                ids=order[start:start+1024]; xb=xt[ids].to(device); yb=yt[ids].to(device)
                loss=F.cross_entropy(model(xb),yb,weight=weights)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        pred=[]
        with torch.no_grad():
            for start in range(0,len(xe),4096): pred.extend(model(xe[start:start+4096]).argmax(-1).cpu().tolist())
        pred=np.asarray(pred)
        record={'seed':seed,'balanced_accuracy':float(balanced_accuracy_score(ye,pred)),'nmi':float(normalized_mutual_info_score(ye,pred)),'ari':float(adjusted_rand_score(ye,pred)),'confusion':pd.crosstab(pd.Series(ye,name='true'),pd.Series(pred,name='pred')).to_dict()}
        records.append(record); print(seed,record['balanced_accuracy'],record['nmi'],flush=True)
    out=HERE/'results'; out.mkdir(parents=True,exist_ok=True); (out/'probe.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__': main()
